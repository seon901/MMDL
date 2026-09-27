#!/usr/bin/env bash
# MMMU-val baseline evaluation 재현 스크립트 (3조 - 20230184 김혜빈)
# Usage: bash run_mmmu_eval_20230184_김혜빈.sh [model_path] [data_root] [output_path]
set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${2:-MMMU/MMMU}"
OUTPUT_PATH="${3:-results/mmmu_baseline_20230184_김혜빈.jsonl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_mmmu_eval_20230184_김혜빈.py" \
  --model_path "${MODEL_PATH}" \
  --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data_root "${DATA_ROOT}" \
  --data_revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \
  --split validation \
  --output_path "${OUTPUT_PATH}" \
  --max_new_tokens 128 \
  --seed 42
