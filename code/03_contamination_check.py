"""
03_contamination_check.py
외부 학습 후보 데이터셋이 MMMU validation과 겹치는지 검사한다.
  - 이미지: perceptual hash(pHash) 해밍 거리
  - 텍스트: MinHash(단어 3-gram) Jaccard 유사도
걸린 쌍은 사람이 눈으로 확인한 뒤 제거한다. 임계값은 휴리스틱이라 결과 분포를 보고 조정한다.
출처(provenance) 확인은 별도로 데이터셋 카드를 읽고 표로 정리한다.

  pip install imagehash datasketch

[실행 예] 컬럼 이름은 데이터셋마다 다르니 데이터셋 카드를 보고 지정한다.
  python3 03_contamination_check.py --cand_name <HF id 또는 로컬 경로> --cand_split train \
      --text_cols question,choices --image_col image --id_col "" --tag scienceqa
"""
import argparse
import ast
import csv
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


def phash_int(img):
    return int(str(imagehash.phash(img.convert("RGB"))), 16)


def min_hamming(ref, h):
    """ref: uint64 배열, h: int. 가장 가까운 ref의 (거리, 인덱스)."""
    x = np.bitwise_xor(ref, np.uint64(h))
    bits = np.unpackbits(x.view(np.uint8)).reshape(len(ref), 64).sum(axis=1)
    i = int(bits.argmin())
    return int(bits[i]), i


def field_to_text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(map(str, v))
    if isinstance(v, str) and v.startswith("["):
        try:
            return " ".join(map(str, ast.literal_eval(v)))
        except (ValueError, SyntaxError):
            pass
    return str(v)


def build_mmmu_index(args):
    hashes, hash_owner = [], []
    lsh = MinHashLSH(threshold=args.text_thresh, num_perm=NUM_PERM)
    mh_store, text_store = {}, {}
    for subj in SUBJECTS:
        ds = load_dataset(args.mmmu_root, subj, split="validation", revision=args.mmmu_revision)
        for ex in ds:
            qid = ex["id"]
            text = ex["question"] + " " + field_to_text(ex["options"])
            sh = shingles(text)
            if len(sh) >= 3:
                m = make_minhash(sh)
                lsh.insert(qid, m)
                mh_store[qid] = m
            text_store[qid] = text
            for k in range(1, 8):
                img = ex.get(f"image_{k}")
                if img is not None:
                    hashes.append(phash_int(img))
                    hash_owner.append(qid)
        print(f"[MMMU 인덱스] {subj} 완료")
    return np.array(hashes, dtype=np.uint64), hash_owner, lsh, mh_store, text_store


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_name", required=True)
    p.add_argument("--cand_config", default=None)
    p.add_argument("--cand_split", default="train")
    p.add_argument("--text_cols", default="question", help="쉼표로 구분")
    p.add_argument("--image_col", default="image", help="이미지 없으면 빈 문자열")
    p.add_argument("--id_col", default="", help="비우면 행 번호 사용")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="cand")
    p.add_argument("--img_thresh", type=int, default=8, help="이 거리 이하면 검토 대상")
    p.add_argument("--text_thresh", type=float, default=0.6)
    p.add_argument("--mmmu_root", default="MMMU/MMMU")
    p.add_argument("--mmmu_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--out_dir", default="contamination_out")
    args = p.parse_args()

    ref_hashes, hash_owner, lsh, mh_store, text_store = build_mmmu_index(args)
    cand = load_dataset(args.cand_name, args.cand_config, split=args.cand_split)
    text_cols = [c for c in args.text_cols.split(",") if c]

    flagged, n_checked = [], 0
    for i, ex in enumerate(cand):
        if args.limit and i >= args.limit:
            break
        n_checked += 1
        cid = str(ex[args.id_col]) if args.id_col else str(i)
        ctext = " ".join(field_to_text(ex.get(c)) for c in text_cols)

        img_hit = None
        img = ex.get(args.image_col) if args.image_col else None
        if img is not None and len(ref_hashes):
            d, j = min_hamming(ref_hashes, phash_int(img))
            if d <= args.img_thresh:
                img_hit = (d, hash_owner[j])

        text_hits = []
        sh = shingles(ctext)
        if len(sh) >= 3:
            m = make_minhash(sh)
            for qid in lsh.query(m):
                text_hits.append((round(m.jaccard(mh_store[qid]), 3), qid))

        if img_hit or text_hits:
            best_text = max(text_hits) if text_hits else None
            mmmu_id = img_hit[1] if img_hit else best_text[1]
            flagged.append({
                "cand_id": cid,
                "mmmu_id": mmmu_id,
                "match": "both" if (img_hit and text_hits) else ("image" if img_hit else "text"),
                "img_hamming": img_hit[0] if img_hit else "",
                "text_jaccard": best_text[0] if best_text else "",
                "cand_text": ctext[:300],
                "mmmu_text": text_store.get(mmmu_id, "")[:300],
                "verdict": "",  # 사람이 채움: same / similar / different
            })
        if n_checked % 5000 == 0:
            print(f"{n_checked}개 검사, {len(flagged)}개 의심")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"flagged_{args.tag}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["cand_id", "mmmu_id", "match", "img_hamming", "text_jaccard",
                  "cand_text", "mmmu_text", "verdict"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flagged)
    print(f"\n검사 {n_checked}개 중 의심 {len(flagged)}개 -> {out}")


if __name__ == "__main__":
    main()
