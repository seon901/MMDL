from pathlib import Path
from datetime import datetime
import hashlib
import importlib.util
import json

import pyarrow.parquet as pq

ROOT = Path("/shared_dir/mmdl")
JOB = Path(__file__).resolve().parent
CURRENT = (
    ROOT / "data/engineering83_prepared_20261118"
    / "validation_20261118_230546_267103"
)
MFR = (
    ROOT / "data/external_candidates/mmfinereason_123k"
    / "cda6140bfba449bef9e57065a528b7424966e144"
)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

spec = importlib.util.spec_from_file_location(
    "previous_audit", JOB / "audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

verification = read(JOB / "file_verification.json")
original = Path(verification["local_file"])

with original.open(encoding="utf-8-sig") as stream:
    first = next(audit.array_items(stream))

human = next(
    turn["value"]
    for turn in first["conversations"]
    if turn["from"] == "human"
)

prefix, marker, remainder = human.partition("<image>")
prefix = prefix.strip()

assert marker, "Expected image marker was not found"
assert prefix.startswith(
    "A conversation between User and Assistant."
), "Unexpected common instruction"
assert prefix.endswith(
    "<answer>final answer here</answer>."
), "Unexpected common instruction ending"

links = {
    item["id"]: item
    for item in read(CURRENT / "source_links.json")
}
items = [
    item
    for item in read(CURRENT.parent / "paired_candidates.json")
    if item["source"] == "MMR1"
]
assert len(items) == 10

hints = read(JOB / "diagnostic_position_hints.json")
official = {}

for hint in hints:
    for text in hint.get("questions", []):
        if not text.startswith(prefix):
            continue

        body = text[len(prefix):]
        key = audit.canonical(body)

        if key:
            official.setdefault(key, []).append({
                "official_zero_based_row": hint["zero_based_row"],
                "official_images": hint.get("images"),
                "official_question": body,
                "official_full_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            })

tables = {}
records = []

for item in items:
    link = links[item["id"]]
    name = link["parquet_file"]

    if name not in tables:
        tables[name] = pq.read_table(
            MFR / name,
            columns=["id", "source", "question", "ori_question"],
        ).to_pylist()

    row = tables[name][link["resolved_zero_based_row"]]

    assert str(row["id"]) == str(item["source_id"])
    assert row["source"] == "MMR1"
    assert row["question"].strip() == item["original_question"].strip()

    matches = []
    for field in ("question", "ori_question"):
        key = audit.canonical(row.get(field))
        if not key:
            continue

        for evidence in official.get(key, []):
            matches.append({
                "matched_candidate_field": field,
                **evidence,
            })

    records.append({
        "candidate_id": item["id"],
        "source_id": item["source_id"],
        "train_question_linked": bool(matches),
        "matches": matches,
    })

matched = sum(record["train_question_linked"] for record in records)
unmatched = [
    record["candidate_id"]
    for record in records
    if not record["train_question_linked"]
]

result = {
    "repository": audit.REPO,
    "revision": audit.REV,
    "matched_candidates": matched,
    "unmatched": unmatched,
    "total_train_question_links": 69 + matched,
    "removed_exact_common_prefix": prefix,
    "normalization": (
        "Remove the exact shared leading instruction, then apply "
        "the previous image-marker and whitespace normalization."
    ),
    "scope": (
        "Train question matches only. Row positions alone were not "
        "accepted. Original image identity and source usage terms "
        "are not established by this check."
    ),
    "input_sha256": {
        "diagnostic_position_hints.json": sha256(
            JOB / "diagnostic_position_hints.json"
        ),
        "source_links.json": sha256(CURRENT / "source_links.json"),
        "paired_candidates.json": sha256(
            CURRENT.parent / "paired_candidates.json"
        ),
        "correction_script": sha256(Path(__file__)),
    },
    "records": records,
}

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
destination = JOB / f"corrected_question_links_{stamp}.json"
destination.write_text(
    json.dumps(result, ensure_ascii=True, indent=2),
    encoding="utf-8",
)

print("=== 공통 지시문 보정 후 대조 결과 ===")
print("MMR1 Train 질문 연결:", matched, "/ 10")
print("전체 Train 질문 연결:", 69 + matched, "/ 79")
print("연결되지 않은 후보:", unmatched)
print("결과 파일:", destination)
print("행 위치만으로 연결을 승인하지 않았습니다.")
