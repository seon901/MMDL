"""
06_structure_probe.py   [진단 전용 - 평가 파이프라인 아님]
공학 이미지(트러스, 회로도 등)에서 '값 계산 없이' 구조(연결 관계)만 설명하게 한다.
실행은 코드가 하고, 출력이 실제 구조와 맞는지는 사람이 review.html을 보며 판정한다.

  - 평가와 같은 해상도(min 256 / max 1024 x28x28), 평가와 같은 엔진 설정
  - 진단은 재현성을 위해 greedy(temperature 0)
  - 문항마다 바로 저장하고, 다시 실행하면 이어서 진행

[실행]
  python3 06_structure_probe.py --failures failures_to_tag_v2.csv --limit 10
[결과]
  analysis_out/structure_probe.csv   (judgement 칸을 엑셀에서 채움)
  analysis_out/structure_review.html (이미지 + 모델 설명을 나란히 보는 페이지)
"""
import argparse
import csv
import html
import os

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
SEED = 3407
KEEP_TYPES = ("Diagrams", "Technical Blueprints", "Sketches", "Geometric")  # 표/그래프는 제외

PROMPT = (
    "Do NOT calculate any values and do NOT answer any question. "
    "Describe only the STRUCTURE shown in the image(s).\n"
    "- Truss or frame: list every joint label, then list every member as (joint - joint), "
    "then the supports and the joint where each load is applied.\n"
    "- Circuit: list every component label with the two nodes or neighbouring components it "
    "connects, state which components are in series and which in parallel, and where the sources are.\n"
    "- Otherwise: list the labelled parts and how they are connected.\n"
    "Write [unclear] for anything you cannot determine. Do not guess."
)
HEADER = ["subject", "id", "img_type", "prompt_tokens", "description", "judgement", "notes"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--failures", required=True)
    p.add_argument("--subjects", nargs="+", default=["Architecture_and_Engineering", "Electronics"])
    p.add_argument("--limit", type=int, default=10, help="과목당 최대 문항 수")
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--out_dir", default="analysis_out")
    return p.parse_args()


def question_images(sample):
    q = sample["question"]
    return [sample[f"image_{i}"] for i in range(1, 8)
            if f"<image {i}>" in q and sample.get(f"image_{i}") is not None]


def write_html(csv_path, html_path, img_dir_rel):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    parts = ["<html><head><meta charset='utf-8'><title>구조 진단 검토</title><style>"
             "body{font-family:sans-serif;max-width:1200px;margin:auto;padding:16px}"
             ".card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:18px 0;display:flex;gap:16px}"
             ".l{flex:1}.r{flex:1}.l img{max-width:100%;border:1px solid #ddd;margin:4px}"
             "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px;max-height:520px;overflow:auto}"
             "</style></head><body><h2>구조 설명 판정</h2>"
             "<p>이미지와 오른쪽 설명을 비교해 CSV의 judgement 칸에 correct / partial / wrong 중 하나를 적는다. "
             "(연결 관계가 모두 맞으면 correct, 일부만 맞으면 partial)</p>"]
    for i, r in enumerate(rows, 1):
        imgs = "".join(f"<img src='{html.escape(img_dir_rel)}/{html.escape(r['id'])}_img{k}.png'>"
                       for k in range(1, 8)
                       if os.path.exists(os.path.join(os.path.dirname(html_path), img_dir_rel,
                                                      f"{r['id']}_img{k}.png")))
        parts.append(f"<div class='card'><div class='l'><h3>#{i} {html.escape(r['id'])}</h3>{imgs}</div>"
                     f"<div class='r'><b>모델 설명</b><pre>{html.escape(r['description'])}</pre></div></div>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    args = parse_args()
    with open(args.failures, encoding="utf-8-sig") as f:
        allrows = list(csv.DictReader(f))
    targets = []
    for s in args.subjects:
        sub = [r for r in allrows if r["subject"] == s and any(k in r["img_type"] for k in KEEP_TYPES)]
        targets += sorted(sub, key=lambda r: r["id"])[: args.limit]
    print(f"대상 {len(targets)}문항")

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "structure_images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "structure_probe.csv")
    html_path = os.path.join(args.out_dir, "structure_review.html")

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig") as f:
            done = {r["id"] for r in csv.DictReader(f)}
        print(f"이어하기: {len(done)}문항 완료됨")
    new_file = not os.path.exists(csv_path)

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    llm = LLM(model=args.model_path, revision=args.model_revision,
              limit_mm_per_prompt={"image": 8}, max_model_len=4096,
              gpu_memory_utilization=0.90, seed=SEED, trust_remote_code=True,
              enforce_eager=True, max_num_seqs=2)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)

    cache = {}
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
            f.flush()
        for idx, r in enumerate(targets, 1):
            if r["id"] in done:
                continue
            subj = r["subject"]
            if subj not in cache:
                if os.path.isdir(args.data_root):
                    ds = load_dataset(args.data_root, subj, split="validation")
                else:
                    ds = load_dataset(args.data_root, subj, split="validation",
                                      revision=args.data_revision)
                cache[subj] = {ex["id"]: ex for ex in ds}
            ex = cache[subj][r["id"]]
            imgs = question_images(ex)
            for k, im in enumerate(imgs, 1):
                im.convert("RGB").save(os.path.join(img_dir, f"{r['id']}_img{k}.png"))
            content = [{"type": "image", "image": im, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
                       for im in imgs]
            content.append({"type": "text", "text": PROMPT})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            vin = {"prompt": text}
            if image_inputs:
                vin["multi_modal_data"] = {"image": image_inputs}
            try:
                out = llm.generate([vin], sp, use_tqdm=False)[0]
                w.writerow([subj, r["id"], r["img_type"], len(out.prompt_token_ids),
                            out.outputs[0].text, "", ""])
            except ValueError:
                w.writerow([subj, r["id"], r["img_type"], "", "", "SKIPPED", "too long"])
            f.flush()
            os.fsync(f.fileno())
            print(f"[{idx}/{len(targets)}] {r['id']} 저장")

    write_html(csv_path, html_path, "structure_images")
    print(f"\n완료: {csv_path}\n검토 페이지: {html_path} (브라우저로 열어서 판정)")


if __name__ == "__main__":
    main()
