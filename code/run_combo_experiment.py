
from pathlib import Path
from datetime import datetime
import fcntl
import json
import subprocess
import sys
import traceback

JOB = Path(__file__).resolve().parent
OLD = Path(
    "/shared_dir/mmdl/checkpoints/"
    "native_reasoning_compare_20261119_232833_892795"
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def status(state, **extra):
    save(JOB / "overall_status.json", {
        "상태": state,
        "갱신 시각": datetime.now().isoformat(),
        **extra,
    })


with (JOB / "orchestrator.lock").open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("같은 조합 실험이 실행 중입니다.")

    try:
        for phase, filename, logfile in [
            ("학습과 이미지 평가", "train_runner.py", "training_stage.log"),
            ("텍스트 평가", "evaluate_text.py", "text_stage.log"),
        ]:
            status(
                "조합 실험 실행 중",
                단계=phase,
                로그=str(JOB / logfile),
            )
            with (JOB / logfile).open("a", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-u", str(JOB / filename)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        combined = read(JOB / "combo_comparison.json")

        old_image = {
            item["구성"]: item["정답 수"]
            for item in read(OLD / "format_comparison.json")["결과"]
        }
        new_image = {
            item["구성"]: item["정답 수"]
            for item in combined["이미지 개발 400개"]
        }
        text = {
            item["구성"]: item["정답 수"]
            for item in combined["텍스트 개발 300개"]
        }

        cells = []
        for replay, form, label in [
            (0, "정답형", "answer_control"),
            (0, "풀이형", "native_reasoning"),
            (20, "정답형", "replay_answer_control"),
            (20, "풀이형", "replay_native_reasoning"),
        ]:
            cells.append({
                "Replay 비율": replay,
                "지도 형식": form,
                "모델": label,
                "이미지 정답 수": (
                    old_image[label] if replay == 0 else new_image[label]
                ),
                "텍스트 정답 수": text[label],
            })

        image_interaction = (
            new_image["replay_native_reasoning"]
            - new_image["replay_answer_control"]
            - old_image["native_reasoning"]
            + old_image["answer_control"]
        )
        text_interaction = (
            text["replay_native_reasoning"]
            - text["replay_answer_control"]
            - text["native_reasoning"]
            + text["answer_control"]
        )

        report = {
            "네 조건 결과": cells,
            "이미지 정답 수 기준 상호작용 관측차": image_interaction,
            "텍스트 정답 수 기준 상호작용 관측차": text_interaction,
            "주의": (
                "단일 시드의 내부 개발 결과입니다. "
                "지도 토큰 수는 형식별로 다릅니다. "
                "최종 MMMU 우월성은 별도 고정 평가가 필요합니다."
            ),
        }

        save(JOB / "factorial_comparison.json", report)
        status(
            "조합 학습과 이미지 및 텍스트 평가 완료",
            결과=report,
            결과파일=str(JOB / "factorial_comparison.json"),
            상세파일=str(JOB / "combo_comparison.json"),
        )
        print(
            json.dumps(
                read(JOB / "overall_status.json"),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    except Exception:
        status(
            "오류로 중단",
            오류=traceback.format_exc(),
            단계=phase,
            로그=str(JOB / logfile),
        )
        raise
