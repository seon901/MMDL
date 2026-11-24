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
ORIGINAL = RUN / "generation_eval_20261119_090625_E0dZcD/evaluate.py"

def build(source):
    class Change(ast.NodeTransformer):
        limits = 0
        models = 0
        settings = 0

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.test, ast.Constant)
                and node.test.value is False
            ):
                return node.orelse
            return node

        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 512:
                self.limits += 1
                return ast.copy_location(
                    ast.Constant(value=2048),
                    node,
                )
            if node.value == "네 모델의 생성 완료":
                return ast.copy_location(
                    ast.Constant(value="두 모델의 생성 완료"),
                    node,
                )
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)

            if any(
                isinstance(target, ast.Name)
                and target.id == "models"
                for target in node.targets
            ):
                value = node.value
                assert isinstance(value, ast.Dict)

                assert {key.value for key in value.keys} == {
                    "base",
                    "recipe_a",
                    "answer_only",
                    "concise_reasoning",
                }, "모델 목록이 예상과 다릅니다."

                pairs = [
                    (key, item)
                    for key, item in zip(value.keys, value.values)
                    if key.value in {"base", "recipe_a"}
                ]
                value.keys = [key for key, item in pairs]
                value.values = [item for key, item in pairs]
                self.models += 1

            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)

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
                    "evaluation_variant": (
                        "original_prompt_2048_tokens_base_and_recipe_a"
                    ),
                    "parent_script": str(ORIGINAL),
                    "parent_script_sha256": hashlib.sha256(
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
        isinstance(node, ast.FunctionDef)
        and node.name == "answer_first_prompt"
        for node in ast.walk(tree)
    ), "원래 풀이 지시가 있는 평가 코드가 아닙니다."

    change = Change()
    tree = change.visit(tree)

    assert (
        change.models == 1
        and change.settings == 1
        and change.limits in {1, 3}
    ), "코드 구조 검사에 실패했습니다. 평가를 시작하지 않습니다."

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
    out = RUN / f"generation_original_2048_{stamp}"
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
            "변경": "출력 한도를 512에서 2048토큰으로 늘렸습니다.",
            "유지": (
                "첫 번째 평가의 질문, 이미지 설정과 "
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

    print("=== 원래 풀이 지시와 2048토큰 생성 평가 실행 요청 완료 ===")
    print("결과 폴더:", out)
    print("프로세스 번호:", process.pid)

if __name__ == "__main__":
    main()
