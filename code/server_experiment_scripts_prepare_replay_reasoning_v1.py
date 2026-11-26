from pathlib import Path
from collections import Counter
from datetime import datetime
import copy
import hashlib
import json

ROOT = Path("/shared_dir/mmdl")
A = ROOT / "data/recipe_a_answer_only_20261115_004649_140669"
FORMAT = ROOT / "data/native_reasoning_pair_20261119_232833_892795"
R20 = (
    ROOT
    / "checkpoints/replay_compare_20261119_214637_363626/data/R20"
)
PROFILE = ROOT / "checkpoints/text_replay_profile_20261119_214637_363626"

TAGS = [
    "scienceqa_train",
    "cosyn_circuit",
    "cosyn_music",
    "cosyn_chemical",
]
QUOTAS = dict(zip(TAGS, [1280, 800, 640, 480]))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(item):
    return (
        item["출처"],
        item["원본 행 번호"],
        item.get("QA 번호"),
    )


assert ROOT.is_dir(), "학교 컨테이너에서 실행해야 합니다."

original = read(A / "train.json")
provenance = read(A / "train_provenance.json")
control = read(FORMAT / "answer_control/train.json")
reasoning = read(FORMAT / "native_reasoning/train.json")

assert (
    len(original)
    == len(provenance)
    == len(control)
    == len(reasoning)
    == 4000
)

summary = read(FORMAT / "preparation_summary.json")

for name in ["answer_control", "native_reasoning"]:
    assert sha(FORMAT / name / "train.json") == (
        summary["학습 데이터 SHA256"][name]
    )

required = {
    index
    for index in range(4000)
    if control[index]["conversations"][-1]["value"]
    != reasoning[index]["conversations"][-1]["value"]
}
assert len(required) == 722
assert all(
    provenance[index]["출처"] == "scienceqa_train"
    for index in required
)

indices = {
    key(item): index
    for index, item in enumerate(provenance)
}
assert len(indices) == 4000

r_rows = read(R20 / "train.json")
r_provenance = read(R20 / "train_provenance.json")
assert len(r_rows) == len(r_provenance) == 4000

preferred = {
    indices[key(item)]
    for row, item in zip(r_rows, r_provenance)
    if row.get("image")
}
text_pairs = [
    (row, item)
    for row, item in zip(r_rows, r_provenance)
    if not row.get("image")
]

assert len(preferred) == 3200
assert len(text_pairs) == 800
assert len({
    item["입력 SHA256"]
    for _, item in text_pairs
}) == 800

text_dev_groups = {
    item["질문 그룹"]
    for item in read(
        PROFILE / "internal_dev_provenance.json"
    )
}
assert not (
    {
        item["질문 그룹"]
        for _, item in text_pairs
    }
    & text_dev_groups
)

chosen = set()

for tag, quota in QUOTAS.items():
    forced = {
        index
        for index in required
        if provenance[index]["출처"] == tag
    }
    candidates = [
        index
        for index in sorted(preferred)
        if provenance[index]["출처"] == tag
        and index not in forced
    ]

    assert len(forced) <= quota
    assert len(candidates) >= quota - len(forced)

    chosen.update(forced)
    chosen.update(candidates[:quota - len(forced)])

assert len(chosen) == 3200
assert required <= chosen

image_indices = sorted(chosen)

control_rows = (
    [copy.deepcopy(control[index]) for index in image_indices]
    + [copy.deepcopy(row) for row, _ in text_pairs]
)
reasoning_rows = (
    [copy.deepcopy(reasoning[index]) for index in image_indices]
    + [copy.deepcopy(row) for row, _ in text_pairs]
)

assert len(control_rows) == len(reasoning_rows) == 4000

changes = 0

for first, second in zip(control_rows, reasoning_rows):
    assert first.get("image") == second.get("image")
    assert first["conversations"][0] == second["conversations"][0]

    gold = first["conversations"][-1]["value"]
    assert second["conversations"][-1]["value"].splitlines()[-1] == gold

    changes += int(
        gold != second["conversations"][-1]["value"]
    )

assert changes == 722

out = ROOT / "data" / (
    "replay_reasoning_pair_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
out.mkdir(exist_ok=False)

hashes = {}

for name, values in [
    ("replay_answer_control", control_rows),
    ("replay_native_reasoning", reasoning_rows),
]:
    folder = out / name
    folder.mkdir()

    (folder / "images").symlink_to(
        A / "images",
        target_is_directory=True,
    )

    save(folder / "train.json", values)

    records = [
        {
            **provenance[index],
            "원래 A 학습 번호": index,
            "풀이형 적용 대상": (
                index in required
                and name == "replay_native_reasoning"
            ),
        }
        for index in image_indices
    ] + [
        copy.deepcopy(item)
        for _, item in text_pairs
    ]
    save(folder / "train_provenance.json", records)

    for filename in [
        "internal_dev.json",
        "internal_dev_provenance.json",
    ]:
        (folder / filename).write_bytes(
            (A / filename).read_bytes()
        )
        assert sha(folder / filename) == sha(A / filename)

        (folder / ("text_" + filename)).write_bytes(
            (PROFILE / filename).read_bytes()
        )

    for row in values:
        if row.get("image"):
            assert (folder / row["image"]).is_file()

    hashes[name] = sha(folder / "train.json")

plan = {
    "목적": "Replay 20% 조건에서 동일 문항의 정답형과 풀이형 비교",
    "기존 A 폴더": str(A),
    "기존 지도 형식 데이터": str(FORMAT),
    "기존 R20 데이터": str(R20),
    "학습 문항 수": 4000,
    "이미지 수": 3200,
    "텍스트 Replay 수": 800,
    "풀이형 적용 문항 수": 722,
    "이미지 출처별 수량": dict(Counter(
        provenance[index]["출처"]
        for index in chosen
    )),
    "텍스트 과목별 수량": dict(Counter(
        item["과목"]
        for _, item in text_pairs
    )),
    "기존 R20와 공통 이미지 수": len(chosen & preferred),
    "지시문": (
        "두 구성에서 동일. "
        "기존 지도 형식 실험의 공통 지시문 유지."
    ),
    "설계": (
        "Replay가 없는 기존 두 구성과 함께 2x2 비교. "
        "풀이 대상 722개 유지."
    ),
    "기존 R20와 차이": (
        "지시문과 일부 이미지 선정이 다릅니다. "
        "기존 R20 점수를 새 대조군 점수로 재사용하지 않습니다."
    ),
    "선별 한계": (
        "원본 설명의 논리와 사실성이 "
        "모두 개별 검증된 상태는 아닙니다."
    ),
    "학습량": (
        "문항 수와 학습 단계는 동일. "
        "풀이형의 지도 토큰 수는 더 많습니다."
    ),
    "학습 데이터 SHA256": hashes,
    "개발 데이터 SHA256": sha(A / "internal_dev.json"),
    "개발 원본 기록 SHA256": sha(
        A / "internal_dev_provenance.json"
    ),
    "상태": (
        "쌍 데이터 준비 완료. "
        "실제 전처리 검사와 학습은 후속 단계."
    ),
}

save(out / "preparation_summary.json", plan)

(
    ROOT / "data/latest_replay_reasoning_pair.txt"
).write_text(str(out), encoding="utf-8")

print("=== Replay와 짧은 풀이 조합 데이터 준비 완료 ===")
print(json.dumps(plan, ensure_ascii=False, indent=2))
print("결과 폴더:", out)
