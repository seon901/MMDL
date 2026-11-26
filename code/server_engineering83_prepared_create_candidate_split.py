from pathlib import Path
from datetime import datetime
from collections import Counter
import copy
import hashlib
import json
import random

BASE = Path(__file__).resolve().parent
CURRENT = BASE / "validation_20261118_230546_267103"
CHECK = CURRENT / "contamination_20261118_231821_678269"
MMR1 = Path(
    "/shared_dir/mmdl/data/source_audit/MMR1"
    "/job_20261119_020317_tk0tt6"
    "/corrected_question_links_20261119_085058_757207.json"
)
SEED = 42

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    item["id"]: item
    for item in read(BASE / "paired_candidates.json")
}
links = {
    item["id"]: item
    for item in read(CURRENT / "source_links.json")
}
groups = read(CHECK / "split_groups.json")
source_check = read(MMR1)

assert source_check["matched_candidates"] == 10
assert source_check["total_train_question_links"] == 79
assert not source_check["unmatched"]
assert not read(CHECK / "mmmu_flags.json")
assert not read(CHECK / "previous_data_flags.json")

datasets = {
    "answer_only": read(CURRENT / "candidate_answer_only.json"),
    "concise_reasoning": read(
        CURRENT / "candidate_concise_reasoning.json"
    ),
}
by_kind = {
    name: {row["id"]: row for row in rows}
    for name, rows in datasets.items()
}
assert all(len(rows) == 79 for rows in datasets.values())
assert all(len(rows) == 79 for rows in by_kind.values())

all_ids = set(by_kind["answer_only"])
assert all_ids == set(by_kind["concise_reasoning"])
assert all_ids == set(metadata) == set(links)
assert len(groups) == 70

members = [
    candidate
    for group in groups
    for candidate in group["members"]
]
assert len(members) == len(set(members)) == 79
assert set(members) == all_ids

group_for = {}
groups_by_source = {}

for group in groups:
    sources = {
        metadata[candidate]["source"]
        for candidate in group["members"]
    }
    assert len(sources) == 1, "Mixed-source group requires review"
    source = next(iter(sources))
    groups_by_source.setdefault(source, []).append(group)
    for candidate in group["members"]:
        group_for[candidate] = group["group_id"]

dev_targets = {
    "BMMR": 9,
    "Zebra-CoT-Physics": 3,
    "MMR1": 2,
    "FineVision-visualwebinstruct(filtered)": 1,
    "Euclid30K": 1,
    "MMK12": 0,
}
assert set(groups_by_source) == set(dev_targets)

rng = random.Random(SEED)
dev_groups = set()

for source in sorted(dev_targets):
    target = dev_targets[source]
    available = sorted(
        groups_by_source[source],
        key=lambda group: group["group_id"],
    )
    rng.shuffle(available)

    possible = {0: []}
    for group in available:
        size = len(group["members"])
        for count, selected in list(possible.items()):
            total = count + size
            if total > target:
                continue
            proposal = selected + [group["group_id"]]
            if (
                total not in possible
                or len(proposal) > len(possible[total])
            ):
                possible[total] = proposal

    assert target in possible, (
        f"Cannot meet development target without splitting groups: {source}"
    )
    dev_groups.update(possible[target])

dev_ids = {
    candidate
    for candidate in all_ids
    if group_for[candidate] in dev_groups
}
train_ids = all_ids - dev_ids

assert len(train_ids) == 63 and len(dev_ids) == 16
assert not train_ids & dev_ids
assert not (
    {group_for[candidate] for candidate in train_ids}
    & {group_for[candidate] for candidate in dev_ids}
)

for candidate in sorted(all_ids):
    answer = by_kind["answer_only"][candidate]
    reasoning = by_kind["concise_reasoning"][candidate]

    assert answer["image"] == reasoning["image"]
    assert answer["conversations"][0] == reasoning["conversations"][0]

    final = "Answer: " + metadata[candidate]["final_answer"]
    assert answer["conversations"][1]["value"].splitlines()[-1] == final
    assert reasoning["conversations"][1]["value"].splitlines()[-1] == final

    image = CURRENT / answer["image"]
    assert image.is_file()
    assert digest(image) == links[candidate]["image_file_sha256"]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output = BASE / f"paired_candidate_split_{stamp}"
output.mkdir(exist_ok=False)

def save(name, value):
    (output / name).write_text(
        json.dumps(value, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

split_ids = {
    "train": sorted(train_ids),
    "internal_dev": sorted(dev_ids),
}

for kind, records in by_kind.items():
    for split, identifiers in split_ids.items():
        rows = []
        for candidate in identifiers:
            row = copy.deepcopy(records[candidate])
            row["image"] = str(
                (CURRENT / row["image"]).resolve()
            )
            rows.append(row)
        save(f"{kind}_{split}.json", rows)

provenance = []
counts = {}

for split, identifiers in split_ids.items():
    counts[split] = dict(Counter(
        metadata[candidate]["source"]
        for candidate in identifiers
    ))
    for candidate in identifiers:
        provenance.append({
            "id": candidate,
            "split": split,
            "group_id": group_for[candidate],
            "source": metadata[candidate]["source"],
            "source_id": metadata[candidate]["source_id"],
        })

save("split_provenance.json", provenance)

manifest = {
    "purpose": "Small paired supervision comparison",
    "status": "Candidate split only; training has not started",
    "seed": SEED,
    "train_questions": len(train_ids),
    "internal_dev_questions": len(dev_ids),
    "source_counts": counts,
    "train_groups": len({
        group_for[candidate] for candidate in train_ids
    }),
    "internal_dev_groups": len(dev_groups),
    "image_root": str(CURRENT),
    "limitations": [
        "A 16-question development set is exploratory.",
        "Train question links do not establish source image identity.",
        "Source usage conditions remain separate from question matching.",
    ],
    "input_sha256": {
        "split_groups.json": digest(CHECK / "split_groups.json"),
        "candidate_answer_only.json": digest(
            CURRENT / "candidate_answer_only.json"
        ),
        "candidate_concise_reasoning.json": digest(
            CURRENT / "candidate_concise_reasoning.json"
        ),
        "paired_candidates.json": digest(BASE / "paired_candidates.json"),
        "corrected_mmr1_links": digest(MMR1),
        "split_script": digest(Path(__file__)),
    },
}
save("manifest.json", manifest)

save("checksums.json", {
    path.name: digest(path)
    for path in sorted(output.glob("*.json"))
})

print("=== 공학 후보 분할 완료 ===")
print("학습 후보:", len(train_ids))
print("개발 후보:", len(dev_ids))
print("학습 그룹:", manifest["train_groups"])
print("개발 그룹:", manifest["internal_dev_groups"])
print("출처별 수량:")
print(json.dumps(counts, ensure_ascii=False, indent=2))
print("정답형과 풀이형의 문항·이미지·분할 일치 확인")
print("학습·개발 간 그룹 겹침 없음")
print("결과 폴더:", output)
