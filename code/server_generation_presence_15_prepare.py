from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import subprocess
import sys

RUN = Path(
    "/shared_dir/mmdl/checkpoints"
    "/engineering_format_pilot_20261119_085822_757298"
)
ORIGINAL = (
    RUN
    / "generation_original_2048_20261119_093456_525103"
    / "evaluate.py"
)

HELPER = '''
from transformers import LogitsProcessor, LogitsProcessorList

class GeneratedPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty=1.5):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] == 0:
            return scores

        result = scores.clone()
        for index in range(generated.shape[0]):
            tokens = generated[index].unique()
            result[index, tokens] -= self.penalty

        return result

def verify_presence_penalty():
    ids = torch.tensor([
        [1, 1, 2, 3, 3],
        [4, 5, 5, 6, 6],
    ])
    scores = torch.zeros((2, 8))
    processor = GeneratedPresencePenalty(3, 1.5)

    expected = scores.clone()
    expected[0, 3] = -1.5
    expected[1, 6] = -1.5

    assert torch.equal(processor(ids, scores), expected)
    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.equal(processor(ids[:, :3], scores), scores)
'''

def build(source):
    class Change(ast.NodeTransformer):
        generated = 0
        settings = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
            ):
                assert not any(
                    item.arg == "logits_processor"
                    for item in node.keywords
                ), "반복 억제 처리가 이미 있습니다."

                expression = ast.parse(
                    "LogitsProcessorList(["
                    "GeneratedPresencePenalty(prompt_length, 1.5)"
                    "])",
                    mode="eval",
                ).body

                node.keywords.append(
                    ast.keyword(
                        arg="logits_processor",
                        value=expression,
                    )
                )
                self.generated += 1

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "save"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "settings.json"
            ):
                value = node.args[1]
                assert isinstance(value, ast.Dict)

                additions = {
                    "presence_penalty": 1.5,
                    "presence_penalty_scope": (
                        "generated_tokens_only_once_per_token"
                    ),
                    "presence_parent_script_sha256": hashlib.sha256(
                        source.encode("utf-8")
                    ).hexdigest(),
                }

                for key, item in additions.items():
                    value.keys.append(ast.Constant(value=key))
                    value.values.append(ast.Constant(value=item))

                self.settings += 1

            return node

    tree = ast.parse(source)

    assert not any(
        isinstance(item, ast.ClassDef)
        and item.name == "GeneratedPresencePenalty"
        for item in ast.walk(tree)
    ), "이미 수정된 코드입니다."

    change = Change()
    tree = change.visit(tree)

    assert change.generated == change.settings == 1, (
        "예상한 코드 구조와 다릅니다."
    )

    main_function = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "main"
    )

    main_function.body.insert(
        0,
        ast.Expr(
            value=ast.Call(
                func=ast.Name(
                    id="verify_presence_penalty",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        ),
    )

    position = tree.body.index(main_function)
    tree.body[position:position] = ast.parse(HELPER).body
    ast.fix_missing_locations(tree)

    result = ast.unparse(tree) + "\n"
    ast.parse(result)

    assert all(
        ord(char) < 128 or 0xAC00 <= ord(char) <= 0xD7A3
        for char in result
    ), "문자 검사에 실패했습니다. 평가를 시작하지 않습니다."

    return result

def main():
    source = ORIGINAL.read_text(encoding="utf-8")
    corrected = build(source)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = RUN / f"generation_presence_15_{stamp}"
    out.mkdir(exist_ok=False)

    (out / "evaluate.py").write_text(
        corrected,
        encoding="utf-8",
    )
    (out / "prepare.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out / "change_record.json").write_text(
        json.dumps({
            "대상": ["base", "recipe_a"],
            "변경": "생성한 토큰에 presence_penalty=1.5를 적용했습니다.",
            "유지": (
                "질문, 이미지 설정, 2048토큰 한도와 "
                "나머지 생성 설정을 유지했습니다."
            ),
            "원본 코드 SHA256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(out / "evaluate.py")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print("=== 반복 억제 생성 평가 실행 요청 완료 ===")
    print("결과 폴더:", out)
    print("프로세스 번호:", process.pid)

if __name__ == "__main__":
    main()
