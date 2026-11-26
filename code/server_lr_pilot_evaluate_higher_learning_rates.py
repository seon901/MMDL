from pathlib import Path
from datetime import datetime
from collections import Counter
import gc
import json
import re
import unicodedata

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

root = Path("/shared_dir/mmdl")
base = root / "models/Qwen3-VL-4B-Instruct"
run = root / "checkpoints/lr_pilot_20261115_002232_993"
data = root / "data/pilot_answer_only_20261115_002019_320870"

rows = json.loads(
    (data / "internal_dev.json").read_text(encoding="utf-8")
)
provenance = json.loads(
    (data / "internal_dev_provenance.json").read_text(encoding="utf-8")
)

assert len(rows) == len(provenance) == 80, \
    "개발 데이터와 원본 기록의 수가 일치하지 않습니다."

experiments = [
    ("lr_3e-4", run / "lr_3e-4"),
    ("lr_5e-4", run / "lr_5e-4"),
    ("lr_1e-3", run / "lr_1e-3"),
]

for name, adapter in experiments:
    if adapter is not None:
        assert (adapter / "adapter_model.safetensors").is_file(), \
            f"어댑터 파일이 없습니다: {adapter}"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output = run / f"generation_eval_{stamp}"
output.mkdir(parents=True, exist_ok=False)

processor = AutoProcessor.from_pretrained(
    base,
    local_files_only=True,
)

image_processor = processor.image_processor
image_processor.min_pixels = 3136
image_processor.max_pixels = 200704

if isinstance(getattr(image_processor, "size", None), dict):
    image_processor.size["shortest_edge"] = 3136
    image_processor.size["longest_edge"] = 200704

def normalize(value):
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().split()).strip()

def extract_answer(value):
    matches = re.findall(
        r"^\s*Answer\s*:\s*(.*?)\s*$",
        value,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return matches[-1].strip() if matches else ""

summaries = []

for name, adapter in experiments:
    print(f"\n=== {name} 모델 로딩 ===", flush=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base,
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

    model = model.to("cuda")
    model.eval()
    model.config.use_cache = True

    results = []
    counts = Counter()
    source_counts = {}
    predictions_path = output / f"{name}_predictions.json"

    for index, (row, original) in enumerate(zip(rows, provenance)):
        prompt = row["conversations"][0]["value"]
        expected_text = row["conversations"][-1]["value"]
        expected = extract_answer(expected_text)

        assert expected, f"{index}번 원본 정답 형식이 잘못됐습니다."
        assert prompt.count("<image>") == 1, \
            f"{index}번 이미지 표기 개수가 잘못됐습니다."

        before, after = prompt.split("<image>")
        image_path = data / row["image"]
        assert image_path.is_file(), f"이미지 파일이 없습니다: {image_path}"

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        content = []
        if before.strip():
            content.append({"type": "text", "text": before})

        content.append({"type": "image", "image": image})

        if after.strip():
            content.append({"type": "text", "text": after})

        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        input_length = inputs["input_ids"].shape[-1]

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

        new_tokens = generated[0, input_length:]
        raw_output = processor.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )
        predicted = extract_answer(raw_output)
        correct = bool(predicted) and (
            normalize(predicted) == normalize(expected)
        )

        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        last_token = int(new_tokens[-1]) if len(new_tokens) else None
        exhausted = (
            len(new_tokens) >= 128 and last_token not in eos_ids
        )

        tag = original["출처"]
        source_counts.setdefault(tag, Counter())

        counts["문항 수"] += 1
        counts["정답 수"] += int(correct)
        counts["답 형식 누락 수"] += int(not predicted)
        counts["길이 한도 도달 수"] += int(exhausted)

        source_counts[tag]["문항 수"] += 1
        source_counts[tag]["정답 수"] += int(correct)

        results.append({
            "번호": index,
            "출처": tag,
            "이미지": row["image"],
            "원본 행 번호": original["원본 행 번호"],
            "QA 번호": original["QA 번호"],
            "기준 정답": expected,
            "추출한 답": predicted,
            "정답 여부": correct,
            "생성 토큰 수": len(new_tokens),
            "길이 한도 도달": exhausted,
            "전체 출력": raw_output,
        })

        if (index + 1) % 10 == 0:
            predictions_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"{name}: {index + 1}/80개 완료, "
                f"현재 정답 {counts['정답 수']}개",
                flush=True,
            )

        del inputs, generated, new_tokens
        image.close()

    summary = {
        "모델": name,
        "문항 수": counts["문항 수"],
        "정답 수": counts["정답 수"],
        "정확도": counts["정답 수"] / counts["문항 수"],
        "답 형식 누락 수": counts["답 형식 누락 수"],
        "길이 한도 도달 수": counts["길이 한도 도달 수"],
        "출처별 결과": {
            tag: {
                "문항 수": values["문항 수"],
                "정답 수": values["정답 수"],
            }
            for tag, values in source_counts.items()
        },
    }
    summaries.append(summary)

    predictions_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

report = {
    "평가 대상": "Internal Dev 80개",
    "평가 설정": {
        "생성 방식": "greedy",
        "최대 생성 토큰 수": 128,
        "최소 이미지 픽셀": 3136,
        "최대 이미지 픽셀": 200704,
        "답 추출": "마지막 Answer 줄",
        "정답 비교": "유니코드·대소문자·공백 정규화 후 일치",
    },
    "결과": summaries,
}

(output / "comparison.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== Internal Dev 생성 평가 완료 ===")
for item in summaries:
    print(
        f"{item['모델']} | "
        f"정답 {item['정답 수']}/{item['문항 수']} | "
        f"정확도 {item['정확도']:.1%} | "
        f"답 형식 누락 {item['답 형식 누락 수']}개 | "
        f"길이 한도 도달 {item['길이 한도 도달 수']}개"
    )
print("결과 폴더:", output)
