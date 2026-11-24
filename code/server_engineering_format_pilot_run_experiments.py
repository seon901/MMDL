
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

run = Path(__file__).resolve().parent
root = Path("/shared_dir/mmdl")
repo = root / "code/Qwen3-VL/qwen-vl-finetune"
data = (
    root / "data/engineering83_prepared_20261118"
    / "paired_candidate_split_20261119_085350_215640"
)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_status(value):
    temporary = run / "status.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(run / "status.json")

def set_argument(arguments, name, value):
    if name in arguments:
        arguments[arguments.index(name) + 1] = str(value)
    else:
        arguments.extend([name, str(value)])

lock = (root / "checkpoints/engineering_format_pilot.lock").open("a")

try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    save_status({
        "상태": "중복 실행으로 중단",
        "안내": "다른 지도 형식 비교 작업이 이미 실행 중입니다.",
        "결과 폴더": str(run),
    })
    raise SystemExit("다른 비교 작업이 이미 실행 중입니다.")

(root / "checkpoints/engineering_format_latest.txt").write_text(
    str(run),
    encoding="utf-8",
)

try:
    environment = os.environ.copy()
    environment.update({
        "PILOT_REPO": str(repo),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "CHECK_DATA": "1",
    })

    versions = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )
    (run / "requirements_training.txt").write_text(
        versions,
        encoding="utf-8",
    )

    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    )
    (run / "code_commit.txt").write_text(
        commit,
        encoding="utf-8",
    )

    results = []

    for kind in ("answer_only", "concise_reasoning"):
        folder = run / kind
        folder.mkdir(exist_ok=False)

        prepared = folder / "data"
        prepared.mkdir()

        shutil.copy2(
            data / f"{kind}_train.json",
            prepared / "train.json",
        )

        # 두 실험의 개발 손실은 같은 정답형 기준으로 측정합니다.
        shutil.copy2(
            data / "answer_only_internal_dev.json",
            prepared / "internal_dev.json",
        )

        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in prepared.glob("*.json")
        }
        (folder / "data_hashes.json").write_text(
            json.dumps(hashes, indent=2),
            encoding="utf-8",
        )

        arguments = (
            run / "source_arguments.txt"
        ).read_text(encoding="utf-8").splitlines()

        changes = {
            "--output_dir": folder,
            "--learning_rate": "2e-4",
            "--num_train_epochs": 3,
            "--max_steps": -1,
            "--eval_strategy": "no",
            "--save_strategy": "epoch",
            "--save_total_limit": 2,
            "--save_only_model": "False",
            "--logging_steps": 1,
            "--seed": 42,
            "--data_seed": 42,
        }

        for name, value in changes.items():
            set_argument(arguments, name, value)

        (folder / "training_arguments.txt").write_text(
            "\n".join(arguments) + "\n",
            encoding="utf-8",
        )

        environment["PILOT_DATA"] = str(prepared)

        save_status({
            "상태": "학습 중",
            "지도 형식": kind,
            "현재 로그": str(folder / "training.log"),
            "결과 폴더": str(run),
        })

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=1",
            str(run / "entry.py"),
            *arguments,
        ]

        with (folder / "training.log").open(
            "w",
            encoding="utf-8",
        ) as log:
            subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        result = read(folder / "pilot_result.json")

        assert result["완료 단계"] == 24, (
            f"예상한 24단계와 다릅니다: {result['완료 단계']}"
        )
        assert (folder / "adapter_model.safetensors").is_file(), (
            f"최종 LoRA 파일이 없습니다: {folder}"
        )

        results.append({
            "지도 형식": kind,
            **result,
        })

    (run / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_status({
        "상태": "두 지도 형식 학습 완료",
        "결과 폴더": str(run),
        "결과": results,
    })

except Exception:
    save_status({
        "상태": "오류로 중단",
        "오류": traceback.format_exc(),
        "결과 폴더": str(run),
    })
    raise
