"""
05_budget_recovery_probe.py   [진단 전용 - 평가 파이프라인 아님]
512토큰 소진으로 틀린 객관식 문항을 두 조건으로 다시 풀어, 병목이 '예산'인지 확인한다.

  조건 A: 원래 프롬프트 + max_tokens 2048   (토큰 한도만 늘림)
  조건 B: 효율 지시 프롬프트 + max_tokens 512 (지시만 바꿈, 한도는 그대로)

결과는 '진단용'으로만 보고한다. 개선 성과로 쓰지 않는다.

[실행]
  python3 05_budget_recovery_probe.py --failures failures_to_tag_v2.csv --limit 60
  (전체 195개 중 객관식 168개. --limit 0 이면 전부)

[출력] analysis_out/budget_recovery.csv, 터미널에 요약
"""
import argparse
import ast
import csv
import os
import re

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

LETTER_POOL = "ABCDEFGHIJKLMNOP"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
SEED = 3407

EFFICIENT_SUFFIX = (
    "\nSolve efficiently. Do not repeat or re-check the same calculation. "
    "Use approximation, substitution of the options, or elimination when it saves steps. "
    "You must finish with the final line in exactly this format:\nAnswer: X"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--failures", required=True)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--out_dir", default="analysis_out")
    return p.parse_args()


def build(sample, suffix=""):
    """평가 스크립트의 answer-last 프롬프트와 동일한 구성 (객관식만)."""
    q = sample["question"]
    imgs = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if f"<image {i}>" in q and img is not None:
            imgs.append(img)
    q_clean = re.sub(r"<image \d+>", "", q).strip()
    opts = sample["options"]
    opts = ast.literal_eval(opts) if isinstance(opts, str) else opts
    letters = LETTER_POOL[: len(opts)]
    opt_lines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
    text = (
        f"Question: {q_clean}\nOptions:\n{opt_lines}\n"
        f"Answer the question. At the very end of your response, "
        f"write your final choice on a new line in exactly this format:\n"
        f"Answer: X\nwhere X is a single option letter."
    ) + suffix
    return text, imgs, letters


def extract(text, letters):
    m = re.search(rf"Answer:\s*[*(\[]*\s*([{letters}])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None  # 대체 파싱은 쓰지 않음: 명시적으로 답했는지 보기 위해


def to_input(text, imgs, processor):
    content = [{"type": "image", "image": im, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
               for im in imgs]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    vin = {"prompt": prompt}
    if image_inputs:
        vin["multi_modal_data"] = {"image": image_inputs}
    return vin


def main():
    args = parse_args()
    with open(args.failures, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r["auto_category"].startswith(("A.", "B."))]  # 객관식 + 512토큰 소진
    if args.limit:
        rows = rows[: args.limit]
    print(f"대상 {len(rows)}문항 (객관식, 512토큰 소진 오답)")

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    llm = LLM(model=args.model_path, revision=args.model_revision,
              limit_mm_per_prompt={"image": 8}, max_model_len=4096,
              gpu_memory_utilization=0.90, seed=SEED, trust_remote_code=True,
              enforce_eager=True, max_num_seqs=4)
    base = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                presence_penalty=1.5, seed=SEED)
    sp_a = SamplingParams(max_tokens=2048, **base)
    sp_b = SamplingParams(max_tokens=512, **base)

    cache = {}
    in_a, in_b, meta = [], [], []
    for r in rows:
        subj = r["subject"]
        if subj not in cache:
            if os.path.isdir(args.data_root):
                ds = load_dataset(args.data_root, subj, split="validation")
            else:
                ds = load_dataset(args.data_root, subj, split="validation",
                                  revision=args.data_revision)
            cache[subj] = {ex["id"]: ex for ex in ds}
        ex = cache[subj][r["id"]]
        ta, imgs, letters = build(ex)
        tb, _, _ = build(ex, EFFICIENT_SUFFIX)
        in_a.append(to_input(ta, imgs, processor))
        in_b.append(to_input(tb, imgs, processor))
        meta.append((r, ex["answer"], letters))

    # max_model_len 4096 안에서 2048 토큰 생성이 가능한지는 프롬프트 길이에 달림.
    # 프롬프트가 너무 길어 실패하는 문항은 건너뛰고 기록한다.
    def run(inputs, sp):
        outs = []
        for vin in inputs:
            try:
                outs.append(llm.generate([vin], sp, use_tqdm=False)[0])
            except ValueError:
                outs.append(None)
        return outs

    out_a = run(in_a, sp_a)
    out_b = run(in_b, sp_b)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "budget_recovery.csv")
    stat = {"A": [0, 0, 0], "B": [0, 0, 0]}  # [시도, Answer줄 있음, 정답]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject", "id", "gold", "A_pred", "A_tokens", "A_correct",
                    "B_pred", "B_tokens", "B_correct"])
        for (r, gold, letters), oa, ob in zip(meta, out_a, out_b):
            row = [r["subject"], r["id"], gold]
            for tag, o in (("A", oa), ("B", ob)):
                if o is None:
                    row += ["SKIP", "", ""]
                    continue
                txt = o.outputs[0].text
                pred = extract(txt, letters)
                ok = int(pred == gold)
                stat[tag][0] += 1
                stat[tag][1] += int(pred is not None)
                stat[tag][2] += ok
                row += [pred, len(o.outputs[0].token_ids), ok]
            w.writerow(row)

    print("\n=== 결과 (진단용, 평가 조건 아님) ===")
    print("기준: 이 문항들은 원래 실행에서 전부 오답(정답 0%)이었음")
    for tag, label in (("A", "A. 토큰 2048"), ("B", "B. 효율 지시 + 512토큰")):
        n, has, ok = stat[tag]
        if n:
            print(f"{label}: 시도 {n} | Answer 줄로 답함 {has} ({has/n:.0%}) | 정답 {ok} ({ok/n:.0%})")
    print(f"상세: {path}")


if __name__ == "__main__":
    main()
