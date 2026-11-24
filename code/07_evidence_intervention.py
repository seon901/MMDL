"""
07_evidence_intervention.py   [진단 전용 - 평가 파이프라인 아님]
같은 문항을 (0) 원래 프롬프트 그대로, (1) 사람이 확인한 시각 증거를 앞에 붙여서
각각 N번 풀게 하고 정답률을 비교한다.
  - 증거를 줘도 안 맞으면: 구조 인식이 아니라 지식/추론 문제일 가능성
  - 증거를 주면 맞으면:   구조 인식(보는 것)이 병목이었을 가능성
평가와 같은 프롬프트·sampling·토큰(512)·해상도를 쓰고, 증거 문장만 다르게 한다.

[입력] evidence_to_fill.csv  (id, evidence 두 칸이 필수. evidence가 빈 행은 건너뜀)
[실행]
  python3 07_evidence_intervention.py --evidence analysis_out/evidence_to_fill.csv --n 5
[결과] analysis_out/evidence_intervention.csv
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
HEADER = ["id", "gold", "n", "base_correct", "base_answered", "evid_correct", "evid_answered",
          "leak_warning"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", required=True)
    p.add_argument("--n", type=int, default=5, help="조건마다 샘플링 횟수")
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--out_dir", default="analysis_out")
    return p.parse_args()


def build(sample, evidence=""):
    """평가 스크립트의 answer-last 객관식 프롬프트와 같은 구성. evidence가 있으면 앞에 붙임."""
    q = sample["question"]
    imgs = [sample[f"image_{i}"] for i in range(1, 8)
            if f"<image {i}>" in q and sample.get(f"image_{i}") is not None]
    q_clean = re.sub(r"<image \d+>", "", q).strip()
    opts = sample["options"]
    opts = ast.literal_eval(opts) if isinstance(opts, str) else opts
    letters = LETTER_POOL[: len(opts)]
    opt_lines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
    body = (f"Question: {q_clean}\nOptions:\n{opt_lines}\n"
            f"Answer the question. At the very end of your response, "
            f"write your final choice on a new line in exactly this format:\n"
            f"Answer: X\nwhere X is a single option letter.")
    if evidence:
        body = ("Verified description of what is visible in the image "
                "(written by a person who checked the image):\n" + evidence.strip() + "\n\n" + body)
    return body, imgs, letters, opts


def extract(text, letters):
    """평가 스크립트의 extract_mcq_answer와 같은 규칙. (답 줄이 있었는지도 반환)"""
    m = re.search(rf"Answer:\s*[*(\[]*\s*([{letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), True
    mm = re.findall(rf"\b([{letters}])\b", text)
    return (mm[-1] if mm else None), False


def leak_check(evidence, opts, gold):
    """증거가 정답 선택지의 글자를 그대로 담고 있으면 경고 (정답 누설 방지)."""
    idx = LETTER_POOL.find(gold)
    warn = []
    if 0 <= idx < len(opts):
        g = str(opts[idx]).strip().lower()
        if len(g) >= 4 and g in evidence.lower():
            warn.append("정답 선택지 문구가 증거에 포함됨")
    if re.search(r"\banswer\b", evidence, re.IGNORECASE):
        warn.append("'answer' 단어 포함")
    return "; ".join(warn)


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
    with open(args.evidence, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("evidence") or "").strip()]
    print(f"증거가 채워진 문항 {len(rows)}개")
    if not rows:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "evidence_intervention.csv")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            done = {r["id"] for r in csv.DictReader(f)}
        print(f"이어하기: {len(done)}문항 완료됨")
    new_file = not os.path.exists(path)

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    llm = LLM(model=args.model_path, revision=args.model_revision,
              limit_mm_per_prompt={"image": 8}, max_model_len=4096,
              gpu_memory_utilization=0.90, seed=SEED, trust_remote_code=True,
              enforce_eager=True, max_num_seqs=4)
    sp = SamplingParams(n=args.n, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                        presence_penalty=1.5, max_tokens=512, seed=SEED)

    cache = {}
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
            f.flush()
        for idx, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            subj = re.sub(r"^validation_", "", r["id"])
            subj = re.sub(r"_\d+$", "", subj)
            if subj not in cache:
                if os.path.isdir(args.data_root):
                    ds = load_dataset(args.data_root, subj, split="validation")
                else:
                    ds = load_dataset(args.data_root, subj, split="validation",
                                      revision=args.data_revision)
                cache[subj] = {ex["id"]: ex for ex in ds}
            ex = cache[subj][r["id"]]
            gold = ex["answer"]
            res = {}
            for tag, ev in (("base", ""), ("evid", r["evidence"])):
                text, imgs, letters, opts = build(ex, ev)
                try:
                    out = llm.generate([to_input(text, imgs, processor)], sp, use_tqdm=False)[0]
                    c = a = 0
                    for o in out.outputs:
                        pred, has_line = extract(o.text, letters)
                        c += int(pred == gold)
                        a += int(has_line)
                    res[tag] = (c, a)
                except ValueError:
                    res[tag] = ("SKIP", "SKIP")
            w.writerow([r["id"], gold, args.n, res["base"][0], res["base"][1],
                        res["evid"][0], res["evid"][1], leak_check(r["evidence"], opts, gold)])
            f.flush()
            os.fsync(f.fileno())
            print(f"[{idx}/{len(rows)}] {r['id']}  증거 없음 {res['base'][0]}/{args.n}"
                  f" -> 증거 있음 {res['evid'][0]}/{args.n}")

    tot_b = tot_e = items = 0
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["base_correct"] == "SKIP" or r["evid_correct"] == "SKIP":
                continue
            items += 1
            tot_b += int(r["base_correct"]) / int(r["n"])
            tot_e += int(r["evid_correct"]) / int(r["n"])
    if items:
        print("\n=== 결과 (진단용, 평가 조건 아님) ===")
        print(f"{items}문항 평균 정답률: 증거 없음 {tot_b/items:.0%} -> 증거 있음 {tot_e/items:.0%}")
        print("주의: 문항 수가 적어 경향으로만 해석. leak_warning이 있는 문항은 증거를 고쳐야 함.")


if __name__ == "__main__":
    main()
