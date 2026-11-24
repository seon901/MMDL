"""
03_contamination_check_v2.py   [GPU 불필요, CPU만 사용]
외부 학습 후보 데이터셋이 MMMU(validation, dev, 선택적으로 test)와 겹치는지 검사한다.
  - 이미지: perceptual hash(pHash) 해밍 거리 (거의 빈 이미지는 건너뜀)
  - 텍스트: MinHash(단어 3-gram) Jaccard 유사도
  - 걸린 쌍을 나란히 보여주는 HTML 검토 페이지를 만들고, 사람이 verdict를 적으면 제외 목록을 만든다.
임계값은 휴리스틱이다. 걸린 쌍은 반드시 사람이 눈으로 확인한다.
출처(provenance) 확인은 코드로 할 수 없으므로 데이터셋 카드를 읽고 provenance_table.csv에 적는다.

[1단계: 검사]
  python3 03_contamination_check_v2.py --cand_name derek-thomas/ScienceQA --cand_split train \
      --text_cols question,choices --image_col image --id_col "" --tag scienceqa_train
  (중첩 컬럼은 점으로: --text_cols qa_pairs.question   /  큰 데이터는: --streaming --limit 10000)
[2단계: 사람이 analysis_out/flagged_<tag>.csv 의 verdict 칸에 same / similar / different 입력]
[3단계: 제외 목록 만들기]
  python3 03_contamination_check_v2.py --apply analysis_out/flagged_<tag>.csv --tag <tag>
"""
import argparse
import ast
import csv
import html
import itertools
import os
import re

import imagehash
import numpy as np
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music",
    "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]
NUM_PERM = 128
STRONG_IMG, STRONG_TXT = 4, 0.8   # 이 이하/이상이면 'strong'


# ---------------- 유틸 ----------------
def normalize(text):
    text = re.sub(r"<image\s*\d+>", " ", str(text))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def shingles(text, n=3):
    words = normalize(text).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def make_minhash(sh):
    m = MinHash(num_perm=NUM_PERM)
    for s in sh:
        m.update(s.encode("utf-8"))
    return m


def field_to_text(v):
    """문자열, 리스트, 딕셔너리, 문자열로 된 리스트를 모두 평평한 텍스트로."""
    if v is None:
        return ""
    if isinstance(v, str):
        if v.startswith("["):
            try:
                return field_to_text(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                pass
        return v
    if isinstance(v, dict):
        return " ".join(field_to_text(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return " ".join(field_to_text(x) for x in v)
    return str(v)


def get_path(ex, path):
    cur = ex
    for key in path.split("."):
        cur = cur[key] if isinstance(cur, dict) else getattr(cur, key)
    return cur


def is_blank(img):
    g = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
    return float(g.std()) < 4.0


def phash_int(img):
    return int(str(imagehash.phash(img.convert("RGB"))), 16)


def min_hamming(ref, h):
    x = np.bitwise_xor(ref, np.uint64(h))
    bits = np.unpackbits(x.view(np.uint8)).reshape(len(ref), 64).sum(axis=1)
    i = int(bits.argmin())
    return int(bits[i]), i


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:80]


def split_of(mmmu_id):
    m = re.match(r"(validation|dev|test)_(.+)_\d+$", mmmu_id)
    return (m.group(1), m.group(2)) if m else (None, None)


# ---------------- MMMU 인덱스 ----------------
def load_mmmu(args, subj, split):
    if os.path.isdir(args.mmmu_root):
        return load_dataset(args.mmmu_root, subj, split=split)
    return load_dataset(args.mmmu_root, subj, split=split, revision=args.mmmu_revision)


def build_mmmu_index(args):
    hashes, owner = [], []
    lsh = MinHashLSH(threshold=args.text_thresh, num_perm=NUM_PERM)
    mh_store, text_store = {}, {}
    for split in args.mmmu_splits:
        for subj in SUBJECTS:
            for ex in load_mmmu(args, subj, split):
                qid = ex["id"]
                text = ex["question"] + " " + field_to_text(ex["options"])
                sh = shingles(text)
                if len(sh) >= 3:
                    m = make_minhash(sh)
                    try:
                        lsh.insert(qid, m)
                    except ValueError:
                        pass
                    mh_store[qid] = m
                text_store[qid] = text
                for k in range(1, 8):
                    img = ex.get(f"image_{k}")
                    if img is not None and not is_blank(img):
                        hashes.append(phash_int(img))
                        owner.append(qid)
        print(f"[MMMU 인덱스] {split} 완료 (이미지 해시 {len(hashes)}개)")
    return np.array(hashes, dtype=np.uint64), owner, lsh, mh_store, text_store


# ---------------- 검토 페이지 ----------------
def write_review(args, flagged, img_dir, rel_dir):
    cache = {}

    def mmmu_images(mid):
        split, subj = split_of(mid)
        if split is None:
            return []
        key = (split, subj)
        if key not in cache:
            cache[key] = {ex["id"]: ex for ex in load_mmmu(args, subj, split)}
        ex = cache[key].get(mid)
        names = []
        if ex is not None:
            for k in range(1, 8):
                im = ex.get(f"image_{k}")
                if im is not None:
                    fn = f"mmmu_{safe(mid)}_img{k}.png"
                    p = os.path.join(img_dir, fn)
                    if not os.path.exists(p):
                        im.convert("RGB").save(p)
                    names.append(fn)
        return names

    parts = ["<html><head><meta charset='utf-8'><title>오염 검사 검토</title><style>"
             "body{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px}"
             ".card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}"
             ".row{display:flex;gap:16px}.col{flex:1}.col img{max-width:100%;max-height:360px;border:1px solid #ddd}"
             ".strong{border-color:#c00}.meta{color:#555;font-size:13px}pre{white-space:pre-wrap;font-size:12px}"
             "</style></head><body>",
             f"<h2>오염 검사 검토: {html.escape(args.tag)} (상위 {len(flagged)}개)</h2>"
             "<p>왼쪽이 후보 데이터, 오른쪽이 MMMU. 같은 문제/같은 그림이면 CSV의 verdict에 same, "
             "비슷한 출처나 변형이면 similar, 상관없으면 different를 적는다.</p>"]
    for i, r in enumerate(flagged, 1):
        cimg = r.get("cand_img", "")
        left = f"<img src='{rel_dir}/{html.escape(cimg)}'>" if cimg else "<i>이미지 없음</i>"
        right = "".join(f"<img src='{rel_dir}/{html.escape(n)}'>" for n in mmmu_images(r["mmmu_id"])) \
            or "<i>이미지 없음</i>"
        parts.append(
            f"<div class='card {'strong' if r['severity'] == 'strong' else ''}'>"
            f"<div class='meta'>#{i} | {r['severity']} | 일치 {r['match']} | 이미지 거리 {r['img_hamming']} | "
            f"텍스트 유사도 {r['text_jaccard']} | 후보 {html.escape(r['cand_id'])} ↔ MMMU {html.escape(r['mmmu_id'])}</div>"
            f"<div class='row'><div class='col'>{left}<pre>{html.escape(r['cand_text'])}</pre></div>"
            f"<div class='col'>{right}<pre>{html.escape(r['mmmu_text'])}</pre></div></div></div>")
    parts.append("</body></html>")
    path = os.path.join(args.out_dir, f"review_{args.tag}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ---------------- 3단계: verdict 반영 ----------------
def apply_verdicts(args):
    with open(args.apply, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"].strip().lower()] = verdicts.get(r["verdict"].strip().lower(), 0) + 1
    drop = sorted({r["cand_id"] for r in rows if r["verdict"].strip().lower() in ("same", "similar")})
    empty = verdicts.get("", 0)
    out = os.path.join(args.out_dir, f"exclude_{args.tag}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(drop) + ("\n" if drop else ""))
    print(f"verdict 집계: {verdicts}")
    if empty:
        print(f"[주의] verdict가 비어 있는 행 {empty}개: 판정 전에는 제외 목록에 포함되지 않음")
    print(f"제외할 후보 id {len(drop)}개 -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_name")
    p.add_argument("--cand_config", default=None)
    p.add_argument("--cand_split", default="train")
    p.add_argument("--text_cols", default="question", help="쉼표로 구분, 중첩은 점(예: qa_pairs.question)")
    p.add_argument("--image_col", default="image", help="이미지가 없으면 빈 문자열")
    p.add_argument("--id_col", default="", help="비우면 행 번호")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N개만(0이면 전체)")
    p.add_argument("--streaming", action="store_true", help="큰 데이터셋을 전부 받지 않고 흘려서 읽기")
    p.add_argument("--tag", default="cand")
    p.add_argument("--img_thresh", type=int, default=8, help="이 거리 이하면 검토 대상")
    p.add_argument("--text_thresh", type=float, default=0.6)
    p.add_argument("--mmmu_splits", nargs="+", default=["validation", "dev"],
                   help="test까지 넣으면 인덱스가 매우 커짐")
    p.add_argument("--mmmu_root", default="MMMU/MMMU")
    p.add_argument("--mmmu_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--max_review", type=int, default=200, help="HTML에 넣을 최대 쌍 수")
    p.add_argument("--out_dir", default="analysis_out")
    p.add_argument("--apply", default=None, help="verdict가 채워진 flagged CSV로 제외 목록 생성")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.apply:
        return apply_verdicts(args)
    if not args.cand_name:
        p.error("--cand_name 이 필요합니다")

    ref_hashes, owner, lsh, mh_store, text_store = build_mmmu_index(args)
    cand = load_dataset(args.cand_name, args.cand_config, split=args.cand_split, streaming=args.streaming)
    it = itertools.islice(cand, args.limit) if args.limit else cand
    text_cols = [c for c in args.text_cols.split(",") if c]

    img_dir = os.path.join(args.out_dir, "flagged_images")
    os.makedirs(img_dir, exist_ok=True)
    flagged, n_checked, n_img, n_saved = [], 0, 0, 0
    for i, ex in enumerate(it):
        n_checked += 1
        cid = str(ex[args.id_col]) if args.id_col else str(i)
        ctext = " ".join(field_to_text(get_path(ex, c)) for c in text_cols)

        img_hit, img_obj = None, None
        raw = ex.get(args.image_col) if args.image_col else None
        imgs = raw if isinstance(raw, list) else ([raw] if raw is not None else [])
        for im in imgs:
            if im is None or is_blank(im):
                continue
            n_img += 1
            if len(ref_hashes):
                d, j = min_hamming(ref_hashes, phash_int(im))
                if d <= args.img_thresh and (img_hit is None or d < img_hit[0]):
                    img_hit, img_obj = (d, owner[j]), im

        text_hits = []
        sh = shingles(ctext)
        if len(sh) >= 3:
            m = make_minhash(sh)
            for qid in lsh.query(m):
                text_hits.append((round(m.jaccard(mh_store[qid]), 3), qid))

        if img_hit or text_hits:
            best_t = max(text_hits) if text_hits else None
            mid = img_hit[1] if img_hit else best_t[1]
            strong = (img_hit and img_hit[0] <= STRONG_IMG) or (best_t and best_t[0] >= STRONG_TXT)
            cimg = ""
            if img_obj is not None and n_saved < args.max_review * 2:
                cimg = f"{safe(args.tag)}_{safe(cid)}.png"
                img_obj.convert("RGB").save(os.path.join(img_dir, cimg))
                n_saved += 1
            flagged.append({
                "cand_id": cid, "mmmu_id": mid,
                "match": "both" if (img_hit and text_hits) else ("image" if img_hit else "text"),
                "img_hamming": img_hit[0] if img_hit else "",
                "text_jaccard": best_t[0] if best_t else "",
                "severity": "strong" if strong else "weak",
                "cand_text": ctext[:300], "mmmu_text": text_store.get(mid, "")[:300],
                "cand_img": cimg, "verdict": "",
            })
        if n_checked % 2000 == 0:
            print(f"{n_checked}개 검사, 의심 {len(flagged)}개")

    flagged.sort(key=lambda r: (r["severity"] != "strong",
                                r["img_hamming"] if r["img_hamming"] != "" else 99))
    fields = ["cand_id", "mmmu_id", "match", "img_hamming", "text_jaccard", "severity",
              "cand_text", "mmmu_text", "verdict"]
    out = os.path.join(args.out_dir, f"flagged_{args.tag}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(flagged)
    review = write_review(args, flagged[: args.max_review], img_dir, "flagged_images") if flagged else None

    n_strong = sum(r["severity"] == "strong" for r in flagged)
    summ = os.path.join(args.out_dir, "contamination_summary.csv")
    new = not os.path.exists(summ)
    with open(summ, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "dataset", "config", "split", "mmmu_splits", "checked", "images_checked",
                        "flagged", "strong", "img_matches", "text_matches"])
        w.writerow([args.tag, args.cand_name, args.cand_config or "", args.cand_split,
                    "+".join(args.mmmu_splits), n_checked, n_img, len(flagged), n_strong,
                    sum(r["match"] in ("image", "both") for r in flagged),
                    sum(r["match"] in ("text", "both") for r in flagged)])
    print(f"\n검사 {n_checked}개(이미지 {n_img}장) 중 의심 {len(flagged)}개 (strong {n_strong}) -> {out}")
    if review:
        print(f"검토 페이지: {review}")
    print(f"누적 요약표: {summ}")
    print("다음: flagged CSV의 verdict를 채운 뒤 --apply 로 제외 목록을 만드세요.")


if __name__ == "__main__":
    main()
