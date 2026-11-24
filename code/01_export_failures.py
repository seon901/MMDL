"""
01_export_failures.py
베이스라인 결과 json에서 오답을 추출해 수동 태깅용 CSV, 오답 이미지 폴더, 요약표를 만든다.
평가 파이프라인은 건드리지 않고, 이미 저장된 결과 파일만 읽는다.

[실행 전 1회] 결과 json의 키 이름을 확인하고 아래 FIELD를 맞춘다.
  python3 -c "import json;d=json.load(open('results_vllm/Music.json'));print(type(d));print(d.keys() if isinstance(d,dict) else d[0].keys())"

[실행]
  python3 01_export_failures.py --results_dir results_vllm --out_dir analysis_out
"""
import argparse
import ast
import csv
import json
import os
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

# ---- 결과 json의 키 이름: 네 파일에 맞게 수정 ----
FIELD = {
    "id": "id",           # 문제 id
    "pred": "pred",       # 파싱된 예측 답 (파싱 실패 시 None 또는 빈 문자열)
    "correct": "correct", # 정답 여부 (True/False)
    "raw": "raw_output",  # 모델 원문 출력
}

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music",
    "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_vllm")
    p.add_argument("--out_dir", default="analysis_out")
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--subjects", nargs="*", default=None, help="비우면 30과목 전체")
    p.add_argument("--max_new_tokens", type=int, default=512, help="평가 때 쓴 값과 동일하게")
    return p.parse_args()


def load_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for k in ("results", "records", "items", "samples"):
        if isinstance(data.get(k), list):
            return data[k]
    raise ValueError(f"{path}: 문제별 결과 리스트를 못 찾음. 최상위 키: {list(data.keys())}")


def parse_img_types(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and v.strip():
        try:
            x = ast.literal_eval(v)
            return [str(t) for t in x] if isinstance(x, list) else [str(x)]
        except (ValueError, SyntaxError):
            return [v]
    return ["(none)"]


def load_subject(args, subject):
    if os.path.isdir(args.data_root):
        return load_dataset(args.data_root, subject, split="validation")
    return load_dataset(args.data_root, subject, split="validation", revision=args.data_revision)


def main():
    args = parse_args()
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, revision=args.model_revision)

    rows = []
    subj_rows = []
    imgtype_stat = defaultdict(lambda: [0, 0])  # [정답 수, 전체 수]

    for subj in (args.subjects or SUBJECTS):
        path = os.path.join(args.results_dir, f"{subj}.json")
        if not os.path.exists(path):
            print(f"[경고] {path} 없음, 건너뜀")
            continue
        recs = load_records(path)
        by_id = {ex["id"]: ex for ex in load_subject(args, subj)}

        n_correct = n_wrong = n_trunc = n_parse = 0
        for r in recs:
            qid = r[FIELD["id"]]
            ex = by_id.get(qid)
            if ex is None:
                print(f"[경고] {qid} 를 데이터셋에서 못 찾음")
                continue
            correct = bool(r.get(FIELD["correct"]))
            raw = r.get(FIELD["raw"]) or ""
            n_tok = len(tok(raw, add_special_tokens=False)["input_ids"])
            truncated = n_tok >= args.max_new_tokens - 2  # 재토큰화 기반 근사치
            pred = r.get(FIELD["pred"])
            parse_fail = pred is None or str(pred).strip() in ("", "None")

            for t in parse_img_types(ex.get("img_type")):
                imgtype_stat[t][0] += int(correct)
                imgtype_stat[t][1] += 1

            if correct:
                n_correct += 1
                continue
            n_wrong += 1
            n_trunc += int(truncated)
            n_parse += int(parse_fail)

            img_files = []
            for k in range(1, 8):
                img = ex.get(f"image_{k}")
                if img is not None:
                    fn = f"{qid}_img{k}.png"
                    img.convert("RGB").save(os.path.join(img_dir, fn))
                    img_files.append(fn)

            rows.append({
                "subject": subj,
                "id": qid,
                "question_type": ex.get("question_type", ""),
                "img_type": "|".join(parse_img_types(ex.get("img_type"))),
                "difficulty": ex.get("topic_difficulty", ""),
                "subfield": ex.get("subfield", ""),
                "gold": ex.get("answer", ""),
                "pred": "" if pred is None else pred,
                "out_tokens": n_tok,
                "truncated": int(truncated),
                "parse_fail": int(parse_fail),
                "question": ex.get("question", ""),
                "options": ex.get("options", ""),
                "raw_output": raw,
                "images": "|".join(img_files),
                # ---- 사람이 채우는 칸 ----
                "failure_type": "",
                "reasoning_subtype": "",
                "near_miss": "",
                "notes": "",
            })

        total = n_correct + n_wrong
        subj_rows.append({
            "subject": subj, "total": total,
            "accuracy": round(n_correct / total, 3) if total else "",
            "wrong": n_wrong, "wrong_truncated": n_trunc, "wrong_parse_fail": n_parse,
        })
        print(f"{subj}: 오답 {n_wrong} (잘림 {n_trunc}, 파싱실패 {n_parse})")

    def write_csv(name, data):
        if not data:
            return
        with open(os.path.join(args.out_dir, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    write_csv("failures_to_tag.csv", rows)
    write_csv("subject_summary.csv", sorted(subj_rows, key=lambda x: x["accuracy"]))
    write_csv("img_type_accuracy.csv", sorted(
        [{"img_type": t, "n": n, "accuracy": round(c / n, 3)} for t, (c, n) in imgtype_stat.items()],
        key=lambda x: x["accuracy"],
    ))
    print(f"\n완료: {args.out_dir}/ 에 failures_to_tag.csv, subject_summary.csv, img_type_accuracy.csv, images/")


if __name__ == "__main__":
    main()
