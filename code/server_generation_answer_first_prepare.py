from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import subprocess
import sys

run = Path(
    "/shared_dir/mmdl/checkpoints"
    "/engineering_format_pilot_20261119_085822_757298"
)
original = run / "generation_eval_20261119_090625_E0dZcD/evaluate.py"
source = original.read_text(encoding="utf-8")
source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

def make_variant(source):
    class Change(ast.NodeTransformer):
        text_changes = 0
        setting_changes = 0

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.test, ast.Constant)
                and node.test.value is False
            ):
                return node.orelse
            return node

        def visit_Dict(self, node):
            node = self.generic_visit(node)

            for index, key in enumerate(node.keys):
                value = node.values[index]

                if (
                    isinstance(key, ast.Constant)
                    and key.value == "text"
                    and isinstance(value, ast.Name)
                    and value.id == "question"
                ):
                    node.values[index] = ast.Call(
                        func=ast.Name(
                            id="answer_first_prompt",
                            ctx=ast.Load(),
                        ),
                        args=[value],
                        keywords=[],
                    )
                    self.text_changes += 1

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
                settings = node.args[1]
                assert isinstance(settings, ast.Dict)

                additions = [
                    (
                        "prompt_variant",
                        "answer_first_then_at_most_two_sentences",
                    ),
                    ("parent_script_sha256", source_hash),
                    ("parent_script", str(original)),
                ]

                for key, value in additions:
                    settings.keys.append(ast.Constant(value=key))
                    settings.values.append(ast.Constant(value=value))

                self.setting_changes += 1

            return node

    tree = ast.parse(source)
    change = Change()
    tree = change.visit(tree)

    assert (
        change.text_changes == 1
        and change.setting_changes == 1
    ), "예상한 평가 코드 구조와 다릅니다."

    helper = ast.parse("""
def answer_first_prompt(question):
    question = question.replace(
        "Solve the problem and end with a line beginning with Answer:.",
        "",
    ).strip()
    return question + (
        "\\nStart with the final answer on the first line "
        "in the format Answer: <final answer>. "
        "Then give at most two sentences of explanation. "
        "Do not repeat the question or your explanation."
    )
""").body[0]

    tree.body.insert(0, helper)
    ast.fix_missing_locations(tree)

    corrected = ast.unparse(tree) + "\n"
    ast.parse(corrected)

    assert all(
        ord(char) < 128 or 0xAC00 <= ord(char) <= 0xD7A3
        for char in corrected
    ), "문자 검사에 실패했습니다. 평가를 시작하지 않습니다."

    return corrected

corrected = make_variant(source)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
out = run / f"generation_answer_first_{stamp}"
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
        "변경": "질문 뒤의 출력 지시만 답을 먼저 쓰는 형식으로 변경했습니다.",
        "유지": "모델, 문항, 이미지 설정, 512토큰 한도, 생성 설정을 유지했습니다.",
        "원본 코드 SHA256": source_hash,
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

print("=== 답을 먼저 쓰는 생성 평가 실행 요청 완료 ===")
print("결과 폴더:", out)
print("프로세스 번호:", process.pid)
