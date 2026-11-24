from pathlib import Path
from collections import Counter
from io import BytesIO
import ast
import json
import os
import re

import imagehash
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from datasketch import MinHash
from huggingface_hub import hf_hub_download

root = Path("/shared_dir/mmdl")
candidate_root = (
    root / "data/engineering_verified_candidates_20261117_210249_113813"
)
check = candidate_root / "contamination_check_20261117_210613_987696"
reference = json.loads(
    (check / "reference_files.json").read_text(encoding="utf-8")
)

assert np.__version__ == "2.2.6", \
    f"검사용 NumPy 버전이 예상과 다릅니다: {np.__version__}"
assert Path(np.__file__).resolve().is_relative_to(
    Path(os.environ["CHECK_DEPS"]).resolve()
), "별도 폴더의 NumPy가 로딩되지 않았습니다."

print("검사용 NumPy:", np.__version__, flush=True)

def make_text_hash(value):
    words = re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
    if len(words) >= 3:
        parts = {
            " ".join(words[index:index + 3])
            for index in range(len(words) - 2)
        }
    else:
        parts = set(words)

    if not parts:
        return None

    result = MinHash(num_perm=128)
    for part in sorted(parts):
        result.update(part.encode("utf-8"))
    return result

def make_image_hash(raw):
    with Image.open(BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return imagehash.phash(image)

candidate_rows = pq.read_table(
    candidate_root / "contamination_candidates.parquet"
).to_pylist()

assert len(candidate_rows) == 6, "검사 후보가 6개가 아닙니다."

candidates = []
for row in candidate_rows:
    candidates.append({
        "id": row["id"],
        "image_hash": make_image_hash(row["image"]["bytes"]),
        "text_hash": make_text_hash(row["question"]),
    })

counts = Counter()
flags = []
control_checked = False

print("=== MMMU Validation·Dev 대조 시작 ===", flush=True)

for index, item in enumerate(reference["기준 파일"], start=1):
    path = hf_hub_download(
        repo_id=reference["기준 데이터"],
        repo_type="dataset",
        revision=reference["고정 revision"],
        filename=item["파일"],
        local_dir=root / "data/contamination_reference/MMMU"
            / reference["고정 revision"],
    )

    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            counts["MMMU 검사 문항"] += 1
            question = str(row.get("question") or "")
            options = row.get("options")

            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError):
                    options = [options]

            if isinstance(options, list):
                question += "\n" + "\n".join(map(str, options))

            text_hash = make_text_hash(question)
            reference_id = str(row.get("id") or counts["MMMU 검사 문항"])

            for candidate in candidates:
                if text_hash is None or candidate["text_hash"] is None:
                    continue

                similarity = candidate["text_hash"].jaccard(text_hash)
                if similarity >= 0.6:
                    flags.append({
                        "검사": "MMMU 텍스트",
                        "후보": candidate["id"],
                        "기준 문항": reference_id,
                        "기준 파일": item["파일"],
                        "유사도": similarity,
                    })

            for column, media in row.items():
                if not re.fullmatch(r"image_\d+", column):
                    continue
                if not isinstance(media, dict) or not media.get("bytes"):
                    continue

                value = make_image_hash(media["bytes"])
                counts["MMMU 검사 이미지"] += 1

                if not control_checked:
                    assert value - value == 0, \
                        "동일 이미지 해시 대조가 실패했습니다."
                    assert text_hash is not None, \
                        "대조용 기준 문항 텍스트가 비어 있습니다."
                    assert text_hash.jaccard(text_hash) == 1.0, \
                        "동일 텍스트 해시 대조가 실패했습니다."
                    control_checked = True

                for candidate in candidates:
                    distance = candidate["image_hash"] - value
                    if distance <= 8:
                        flags.append({
                            "검사": "MMMU 이미지",
                            "후보": candidate["id"],
                            "기준 문항": reference_id,
                            "기준 파일": item["파일"],
                            "이미지 컬럼": column,
                            "해밍 거리": int(distance),
                        })

    print(
        f"MMMU 기준 파일 {index}/{len(reference['기준 파일'])}개 완료",
        flush=True,
    )

assert control_checked, "기준 이미지의 정상 처리 여부를 확인하지 못했습니다."

print("=== 기존 Train·Dev 이미지 대조 시작 ===", flush=True)

previous = root / "data/recipe_a_answer_only_20261115_004649_140669"
previous_matches = []
previous_counts = Counter()

for split in ("train", "internal_dev"):
    rows = json.loads(
        (previous / f"{split}.json").read_text(encoding="utf-8")
    )

    for row in rows:
        path = previous / row["image"]
        value = make_image_hash(path.read_bytes())
        previous_counts[split] += 1

        for candidate in candidates:
            distance = candidate["image_hash"] - value
            if distance <= 4:
                previous_matches.append({
                    "검사": "기존 데이터 이미지",
                    "후보": candidate["id"],
                    "분할": split,
                    "기존 이미지": row["image"],
                    "해밍 거리": int(distance),
                })

    print(
        f"{split}: 이미지 {previous_counts[split]}개 대조 완료",
        flush=True,
    )

(check / "mmmu_flags.json").write_text(
    json.dumps(flags, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(check / "previous_data_flags.json").write_text(
    json.dumps(previous_matches, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary = {
    "후보 수": len(candidates),
    "MMMU 검사 집계": dict(counts),
    "MMMU 의심 쌍 수": len(flags),
    "MMMU 의심 후보": sorted({
        item["후보"] for item in flags
    }),
    "기존 데이터 검사 이미지 수": dict(previous_counts),
    "기존 데이터 유사 이미지 쌍 수": len(previous_matches),
    "기존 데이터 유사 이미지 후보": sorted({
        item["후보"] for item in previous_matches
    }),
    "이미지 검사": "64비트 pHash",
    "MMMU 이미지 검토 기준": "해밍 거리 8 이하",
    "텍스트 검사": "단어 3-gram, 128개 순열의 MinHash",
    "MMMU 텍스트 검토 기준": "추정 유사도 0.6 이상",
    "동일 입력 대조 확인": control_checked,
    "주의": (
        "검출된 쌍은 중복 의심이며 수동 대조로 판정해야 합니다. "
        "검출되지 않아도 모든 변형과 원출처 중복을 배제하는 것은 아닙니다."
    ),
}

(check / "check_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== 공학 후보 오염·중복 검사 완료 ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))

if flags:
    print("\nMMMU 의심 쌍 예시:")
    print(json.dumps(flags[:10], ensure_ascii=False, indent=2))

if previous_matches:
    print("\n기존 데이터 유사 이미지 예시:")
    print(json.dumps(previous_matches[:10], ensure_ascii=False, indent=2))

print("검사 결과 폴더:", check)
