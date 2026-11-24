"""
02_perception_probe.py  [진단 전용 - 평가 파이프라인 아님]
문제를 풀지 말고 이미지 속 글자/숫자/기호만 전사하게 해서
"정보를 못 읽어서 틀렸나" vs "읽고도 추론을 못했나"를 분리한다.

  --setting eval  : 평가와 같은 해상도 (min 256 / max 1024 x28x28)  <- 핵심 근거
  --setting hires : 진단용 고해상도 (min 1280 / max 5120 x28x28)  <- 반드시 '진단용'으로 표기

[실행 예]
  python3 02_perception_probe.py --failures analysis_out/failures_to_tag.csv \
      --subjects Music Architecture_and_Engineering Electronics --setting eval
  python3 02_perception_probe.py --failures analysis_out/failures_to_tag.csv \
      --subjects Music Architecture_and_Engineering Electronics --setting hires --max_model_len 6144

이미지 전처리(메시지에 min/max_pixels 지정 -> process_vision_info)는 eval_mmmu_vllm.py의
build_vllm_input과 같은 방식이다.
"""
import argparse
import csv
import os

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SETTINGS = {
    "eval": (256 * 28 * 28, 1024 * 28 * 28),
    "hires": (1280 * 28 * 28, 5120 * 28 * 28),
}

PROBE_PROMPT = (
    "Do NOT solve or answer any question. Only transcribe what is visible in the image(s).\n"
    "List in reading order: every word, number, unit, symbol, axis label, table cell and labeled value.\n"
    "For sheet music: clef, key signature, time signature, then each note as pitch+octave+duration, measure by measure.\n"
    "For circuits or engineering drawings: each component or member label with its value and unit.\n"
    "Write [unreadable] for anything you cannot read clearly. Do not guess."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--failures", required=True, help="01 스크립트가 만든 failures_to_tag.csv")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0, help="과목당 최대 문항 수 (0이면 전부)")
    p.add_argument("--setting", choices=list(SETTINGS), default="eval")
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--max_model_len", type=int, default=4096, help="eval은 4096(평가와 동일), hires는 6144 권장")
    p.add_argument("--out_dir", default="analysis_out")
    return p.parse_args()


def main():
    args = parse_args()
    min_px, max_px = SETTINGS[args.setting]

    with open(args.failures, encoding="utf-8-sig") as f:
        targets = [r for r in csv.DictReader(f)
                   if not args.subjects or r["subject"] in args.subjects]
    if args.limit:
        per = {}
        kept = []
        for r in targets:
            per[r["subject"]] = per.get(r["subject"], 0) + 1
            if per[r["subject"]] <= args.limit:
                kept.append(r)
        targets = kept
    print(f"대상 {len(targets)}문항, setting={args.setting}")

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    # eval_mmmu_vllm.py와 같은 엔진 설정 (enforce_eager: 이전에 겪은 컴파일 오류 회피)
    llm = LLM(model=args.model_path, revision=args.model_revision,
              max_model_len=args.max_model_len, max_num_seqs=2,
              limit_mm_per_prompt={"image": 8}, gpu_memory_utilization=0.90,
              seed=3407, trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)  # 진단은 greedy로 고정

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"probe_{args.setting}.csv")
    cache = {}
    n_ok = n_skip = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject", "id", "img_type", "setting", "prompt_tokens",
                    "transcription", "read_judgement", "notes"])
        # 문항을 하나씩 처리: 너무 긴 문항 하나 때문에 전체가 멈추지 않도록
        for idx, r in enumerate(targets, 1):
            subj = r["subject"]
            if subj not in cache:
                if os.path.isdir(args.data_root):
                    ds = load_dataset(args.data_root, subj, split="validation")
                else:
                    ds = load_dataset(args.data_root, subj, split="validation",
                                      revision=args.data_revision)
                cache[subj] = {ex["id"]: ex for ex in ds}
            ex = cache[subj][r["id"]]

            content = []
            for k in range(1, 8):
                img = ex.get(f"image_{k}")
                if img is not None:
                    content.append({"type": "image", "image": img,
                                    "min_pixels": min_px, "max_pixels": max_px})
            content.append({"type": "text", "text": PROBE_PROMPT})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            try:
                out = llm.generate([{"prompt": text,
                                     "multi_modal_data": {"image": image_inputs}}],
                                   sp, use_tqdm=False)[0]
                w.writerow([subj, r["id"], r["img_type"], args.setting,
                            len(out.prompt_token_ids), out.outputs[0].text, "", ""])
                n_ok += 1
            except ValueError as e:  # 프롬프트가 max_model_len 초과 등
                w.writerow([subj, r["id"], r["img_type"], args.setting, "",
                            "", "SKIPPED", f"too long for max_model_len: {str(e)[:120]}"])
                n_skip += 1
            f.flush()
            print(f"[{idx}/{len(targets)}] {r['id']} 완료")
    print(f"성공 {n_ok}, 건너뜀 {n_skip}")
    print(f"완료: {out_path}  (read_judgement 칸은 사람이 이미지와 비교해 채움)")


if __name__ == "__main__":
    main()
