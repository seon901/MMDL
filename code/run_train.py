"""Run the recorded R20 entry with explicit local paths. No training runs on import."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="저장된 R20 학습 설정 실행")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data = Path(args.data).expanduser().resolve()
    arguments = (repo / "results/r20_training_arguments.txt").read_text(encoding="utf-8").splitlines()
    arguments = [line.strip() for line in arguments if line.strip()]
    for key, value in [("--model_name_or_path", args.model), ("--output_dir", str(Path(args.output).expanduser().resolve()))]:
        arguments[arguments.index(key) + 1] = value
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=1", str(repo / "code/train.py"), *arguments]
    environment = os.environ.copy()
    environment.update(PILOT_REPO=str(repo / "code"), PILOT_DATA=str(data), CHECK_DATA="1", TOKENIZERS_PARALLELISM="false")
    if args.dry_run:
        print(json.dumps({"command": command, "PILOT_REPO": environment["PILOT_REPO"], "PILOT_DATA": environment["PILOT_DATA"]}, ensure_ascii=False, indent=2))
        return
    for filename, expected in [("train.json", 4000), ("internal_dev.json", 400)]:
        path = data / filename
        if not path.is_file():
            parser.error("필요한 데이터가 없습니다: " + str(path))
        if len(json.loads(path.read_text(encoding="utf-8"))) != expected:
            parser.error(filename + " 문항 수가 기록과 다릅니다.")
    subprocess.run(command, cwd=repo / "code", env=environment, check=True)

if __name__ == "__main__":
    main()
