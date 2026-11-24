
from pathlib import Path
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
data = root / "data/recipe_a_answer_only_20261115_004649_140669"
previous = root / "checkpoints/lr_pilot_20261115_002232_993"

environment = os.environ.copy()
environment.update({
    "PILOT_REPO": str(repo),
    "PILOT_DATA": str(data),
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "CHECK_DATA": "0",
})

def write_status(value):
    temporary = run / "status.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(run / "status.json")

def set_argument(arguments, name, value):
    if name in arguments:
        index = arguments.index(name)
        arguments[index + 1] = str(value)
    else:
        arguments.extend([name, str(value)])

try:
    shutil.copy2(
        data / "dataset_manifest.json",
        run / "dataset_manifest.json",
    )
    shutil.copy2(
        data / "token_validation_20261115_010527_624791.json",
        run / "token_validation.json",
    )

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
    (run / "code_commit.txt").write_text(commit, encoding="utf-8")

    data_hashes = {}
    for filename in ("train.json", "internal_dev.json"):
        data_hashes[filename] = hashlib.sha256(
            (data / filename).read_bytes()
        ).hexdigest()

    (run / "data_hashes.json").write_text(
        json.dumps(data_hashes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = []

    for rate in ("1e-4", "2e-4"):
        folder = run / f"lr_{rate}"
        folder.mkdir(exist_ok=False)

        arguments = (
            previous / "lr_1e-4/training_arguments.txt"
        ).read_text(encoding="utf-8").splitlines()

        changes = {
            "--output_dir": folder,
            "--learning_rate": rate,
            "--max_steps": 500,
            "--eval_strategy": "steps",
            "--eval_steps": 100,
            "--save_strategy": "steps",
            "--save_steps": 100,
            "--save_total_limit": 5,
            "--logging_steps": 10,
            "--per_device_eval_batch_size": 1,
        }

        for name, value in changes.items():
            set_argument(arguments, name, value)

        (folder / "training_arguments.txt").write_text(
            "\n".join(arguments) + "\n",
            encoding="utf-8",
        )

        write_status({
            "상태": "학습 중",
            "현재 학습률": rate,
            "전체 결과 폴더": str(run),
            "현재 로그": str(folder / "training.log"),
        })

        print(f"\n=== 학습률 {rate}, 500단계 학습 시작 ===", flush=True)

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=1",
            str(run / "expanded_entry.py"),
            *arguments,
        ]

        with (folder / "training.log").open(
            "w", encoding="utf-8"
        ) as log:
            subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        result_path = folder / "pilot_result.json"
        assert result_path.is_file(), \
            f"실험 결과 파일이 없습니다: {result_path}"

        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        assert result["완료 단계"] == 500, \
            f"{rate} 실험이 500단계까지 완료되지 않았습니다."
        assert (folder / "adapter_model.safetensors").is_file(), \
            f"{rate} 실험의 최종 어댑터가 없습니다."

        results.append(result)
        print(
            json.dumps(result, ensure_ascii=False, indent=2),
            flush=True,
        )

    (run / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_status({
        "상태": "두 학습률 실험 완료",
        "전체 결과 폴더": str(run),
        "결과": results,
    })

    print("\n=== 확대 학습 두 개 완료 ===", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

except Exception:
    error = traceback.format_exc()
    write_status({
        "상태": "오류로 중단",
        "전체 결과 폴더": str(run),
        "오류": error,
    })
    print(error, flush=True)
    raise
