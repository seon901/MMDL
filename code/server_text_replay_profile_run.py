from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import fcntl
import gc
import hashlib
import json
import os
import re
import traceback
import unicodedata

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

ROOT = Path("/shared_dir/mmdl")
JOB = Path(__file__).resolve().parent
SOURCE = ROOT / "data/text_replay_candidates_20261119_214637_363626"
BASE = ROOT / "models/Qwen3-VL-4B-Instruct"
ADAPTER = (
    ROOT
    / "checkpoints/expanded_lr_20261116_223345_306888"
    / "lr_2e-4"
)
SUBJECTS = [
    "language science",
    "natural science",
    "social science",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def status(state, **extra):
    save(JOB / "status.json", {
        "상태": state,
        "갱신 시각": datetime.now().isoformat(),
        **extra,
    })


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def answer(text):
    matches = re.findall(
        r"^\s*Answer\s*:\s*(.*?)\s*$",
        text,
        flags=re.I | re.M,
    )
    return matches[-1].strip() if matches else ""


def main():
    rows = read(SOURCE / "candidates.json")
    provenance = read(SOURCE / "provenance.json")

    assert len(rows) == len(provenance) == 5792
    assert (SOURCE / "compatibility_result.json").is_file()

    hashes = {
        name: hashlib.sha256(
            (SOURCE / name).read_bytes()
        ).hexdigest()
        for name in ["candidates.json", "provenance.json"]
    }

    hash_path = JOB / "source_hashes.json"
    if hash_path.is_file():
        assert read(hash_path) == hashes, (
            "후보 데이터가 변경됐습니다."
        )
    else:
        save(hash_path, hashes)

    status("텍스트 질문 그룹과 개발 분할 준비 중")

    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left = find(left)
        right = find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    exact = {}
    inverted = defaultdict(set)
    grams_by_index = {}

    for index, item in enumerate(provenance):
        text = normalized(item["질문"])

        if text in exact:
            union(index, exact[text])
            continue

        exact[text] = index
        tokens = text.split()
        grams = set(zip(tokens, tokens[1:], tokens[2:]))

        if len(grams) < 10:
            continue

        shared = Counter()
        for gram in grams:
            shared.update(inverted.get(gram, ()))

        for other, overlap in shared.items():
            other_grams = grams_by_index[other]
            similarity = overlap / (
                len(grams) + len(other_grams) - overlap
            )
            if similarity >= 0.8:
                union(index, other)

        grams_by_index[index] = grams
        for gram in grams:
            inverted[gram].add(index)

    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)

    ordered = sorted(
        groups.values(),
        key=lambda indices: fingerprint(
            "42:"
            + min(
                provenance[index]["입력 SHA256"]
                for index in indices
            )
        ),
    )

    dev_counts = Counter()
    dev_indices = []

    for indices in ordered:
        counts = Counter(
            provenance[index]["과목"]
            for index in indices
        )
        if all(
            tag in SUBJECTS
            and dev_counts[tag] + count <= 100
            for tag, count in counts.items()
        ):
            dev_indices.extend(indices)
            dev_counts.update(counts)

    assert all(
        dev_counts[tag] >= 80
        for tag in SUBJECTS
    ), f"개발 분할 과목 수량이 부족합니다: {dict(dev_counts)}"
    assert len(dev_indices) >= 270

    dev_groups = {
        find(index)
        for index in dev_indices
    }
    pools = {tag: [] for tag in SUBJECTS}

    for index, item in enumerate(provenance):
        if find(index) not in dev_groups:
            pools[item["과목"]].append(index)

    for tag in SUBJECTS:
        pools[tag].sort(
            key=lambda index: fingerprint(
                "42:" + provenance[index]["입력 SHA256"]
            )
        )

    pool_indices = []
    cursor = Counter()

    while len(pool_indices) < 1600:
        added = False

        for tag in SUBJECTS:
            if (
                cursor[tag] < len(pools[tag])
                and len(pool_indices) < 1600
            ):
                pool_indices.append(
                    pools[tag][cursor[tag]]
                )
                cursor[tag] += 1
                added = True

        assert added, "학습 선별 후보가 부족합니다."

    dev_indices.sort(
        key=lambda index: provenance[index]["입력 SHA256"]
    )

    assert not (
        {find(index) for index in pool_indices}
        & dev_groups
    )

    split = {
        "train_pool_indices": pool_indices,
        "dev_indices": dev_indices,
        "dev_counts": dict(dev_counts),
        "train_pool_counts": dict(cursor),
        "seed": 42,
        "question_groups": len(groups),
    }

    split_path = JOB / "split_manifest.json"
    if split_path.is_file():
        assert read(split_path) == split
    else:
        save(split_path, split)

    for name, indices in [
        ("train_pool", pool_indices),
        ("internal_dev", dev_indices),
    ]:
        save(
            JOB / f"{name}.json",
            [rows[index] for index in indices],
        )
        save(
            JOB / f"{name}_provenance.json",
            [
                {
                    **provenance[index],
                    "질문 그룹": find(index),
                }
                for index in indices
            ],
        )

    (
        ROOT / "checkpoints/latest_text_replay_profile.txt"
    ).write_text(str(JOB), encoding="utf-8")

    processor = AutoProcessor.from_pretrained(
        BASE,
        local_files_only=True,
    )
    results = {}

    for label, adapter in [
        ("base", None),
        ("A", ADAPTER),
    ]:
        tasks = [("internal_dev", dev_indices)]
        if label == "base":
            tasks.insert(0, ("train_pool", pool_indices))

        needed = []

        for split_name, indices in tasks:
            path = JOB / f"{label}_{split_name}_predictions.json"
            existing = read(path) if path.is_file() else []

            assert len(existing) <= len(indices)

            for position, item in enumerate(existing):
                assert item["input_sha256"] == (
                    provenance[
                        indices[position]
                    ]["입력 SHA256"]
                )

            results[(label, split_name)] = existing

            if len(existing) < len(indices):
                needed.append((split_name, indices))

        if not needed:
            continue

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        status("모델 로딩 중", 모델=label)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        if adapter is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter,
                is_trainable=False,
                local_files_only=True,
            )

        model = model.to("cuda").eval()
        model.config.use_cache = True

        for split_name, indices in needed:
            values = results[(label, split_name)]
            path = JOB / f"{label}_{split_name}_predictions.json"

            for position in range(len(values), len(indices)):
                index = indices[position]
                row = rows[index]

                prompt = row["conversations"][0]["value"]
                gold = answer(
                    row["conversations"][-1]["value"]
                )
                assert gold in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt,
                    }],
                }]

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")

                assert (
                    inputs["input_ids"].shape[-1] + 128 <= 2048
                ), f"문항 길이 초과: {index}"

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        num_beams=1,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )

                new_tokens = generated[
                    0,
                    inputs["input_ids"].shape[-1]:,
                ]
                raw = processor.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                predicted = answer(raw)

                eos = model.generation_config.eos_token_id
                eos_ids = set(
                    eos if isinstance(eos, list) else [eos]
                )
                exhausted = (
                    len(new_tokens) >= 128
                    and int(new_tokens[-1]) not in eos_ids
                )

                values.append({
                    "candidate_index": index,
                    "input_sha256": provenance[index]["입력 SHA256"],
                    "subject": provenance[index]["과목"],
                    "gold": gold,
                    "predicted": predicted,
                    "correct": predicted.casefold() == gold.casefold(),
                    "raw": raw,
                    "tokens": len(new_tokens),
                    "exhausted": exhausted,
                })

                del inputs, generated, new_tokens

                if (
                    (position + 1) % 10 == 0
                    or position + 1 == len(indices)
                ):
                    save(path, values)
                    status(
                        "텍스트 생성 평가 중",
                        모델=label,
                        분할=split_name,
                        완료=position + 1,
                        전체=len(indices),
                    )
                    print(
                        f"{label} {split_name}: "
                        f"{position + 1}/{len(indices)}",
                        flush=True,
                    )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    selected = [
        item["candidate_index"]
        for item in results[("base", "train_pool")]
        if item["correct"] and not item["exhausted"]
    ]

    save(
        JOB / "base_correct_candidates.json",
        [rows[index] for index in selected],
    )
    save(
        JOB / "base_correct_provenance.json",
        [
            {
                **provenance[index],
                "질문 그룹": find(index),
            }
            for index in selected
        ],
    )

    summary = {}

    for label in ["base", "A"]:
        values = results[(label, "internal_dev")]
        summary[label] = {
            "문항 수": len(values),
            "정답 수": sum(item["correct"] for item in values),
            "답 형식 누락": sum(
                not item["predicted"] for item in values
            ),
            "길이 한도 도달": sum(
                item["exhausted"] for item in values
            ),
        }

    base_dev = results[("base", "internal_dev")]
    a_dev = results[("A", "internal_dev")]

    paired = {
        "회복": sum(
            not base["correct"] and trained["correct"]
            for base, trained in zip(base_dev, a_dev)
        ),
        "퇴보": sum(
            base["correct"] and not trained["correct"]
            for base, trained in zip(base_dev, a_dev)
        ),
    }

    save(JOB / "profile_summary.json", {
        "텍스트 개발 결과": summary,
        "A의 회복과 퇴보": paired,
        "선별한 순정 정답 학습 후보": len(selected),
        "선별 후보 과목별 수량": dict(Counter(
            provenance[index]["과목"]
            for index in selected
        )),
        "분할": split,
        "범위": (
            "ScienceQA 텍스트 문항의 유지 평가입니다. "
            "전체 MMMU 분야의 보존을 뜻하지 않습니다."
        ),
    })

    status(
        "순정 정답 후보 선별과 텍스트 개발 평가 완료",
        결과파일=str(JOB / "profile_summary.json"),
        개발결과=summary,
        회복과퇴보=paired,
        학습후보수=len(selected),
    )

    print(
        json.dumps(
            read(JOB / "status.json"),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with (
        ROOT / "checkpoints/rank_comparison.lock"
    ).open("a") as lock:
        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit(
                "다른 비교 작업이 실행 중입니다. "
                "중복 실행을 중단합니다."
            )

        try:
            main()
        except Exception:
            status(
                "오류로 중단",
                오류=traceback.format_exc(),
            )
            raise
