from pathlib import Path
import fcntl, hashlib, json, traceback
JOB = Path(__file__).resolve().parent
RATES = {'scale_110': 1.1, 'scale_125': 1.25, 'scale_150': 1.5}
REFERENCE = Path('/shared_dir/mmdl/checkpoints/expanded_lr_20261116_223345_306888/generation_eval_20261116_231128_946555')
DATA = Path("/shared_dir/mmdl/data/recipe_a_answer_only_20261115_004649_140669")
_checked = False

def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

def _status(**value):
    _save(JOB / "status.json", value)

def _set_scale(model, value):
    from peft.tuners.lora.layer import LoraLayer

    active = model.active_adapters
    assert len(active) == 1
    adapter = active[0]
    assert not getattr(
        model.peft_config[adapter], "use_rslora", False
    )

    count = 0
    for module_name, layer in model.named_modules():
        if isinstance(layer, LoraLayer) and adapter in layer.scaling:
            assert "language_model" in module_name
            assert "visual" not in module_name

            layer.set_scale(adapter, value)
            expected = (
                value * layer.lora_alpha[adapter] / layer.r[adapter]
            )
            assert abs(layer.scaling[adapter] - expected) < 1e-9
            count += 1

    assert count == 144, (
        f"LoRA 대상 모듈 수가 다릅니다: {count}"
    )

def _configure(model, name):
    global _checked

    if not _checked:
        _status(
            상태="기준 결과 재현 확인 중",
            결과폴더=str(JOB),
        )

        groups = {}
        for index, original in enumerate(provenance):
            groups.setdefault(original["출처"], []).append(index)

        indices = [
            index
            for tag in sorted(groups)
            for index in groups[tag][:2]
        ]
        assert len(indices) == 8

        checks = []
        _set_scale(model, 1.0)

        for label in ["base", "lr_2e-4"]:
            for index in indices:
                if label == "base":
                    with model.disable_adapter():
                        prediction, correct = _probe(
                            model,
                            processor,
                            rows[index],
                            provenance[index],
                            index,
                        )
                else:
                    prediction, correct = _probe(
                        model,
                        processor,
                        rows[index],
                        provenance[index],
                        index,
                    )

                cached = _cached[label][index]
                matches = (
                    normalize(prediction)
                    == normalize(cached["추출한 답"])
                    and correct == cached["정답 여부"]
                )
                checks.append({
                    "모델": label,
                    "번호": index,
                    "일치": matches,
                    "기존 답": cached["추출한 답"],
                    "현재 답": prediction,
                })

        _save(JOB / "reference_check.json", checks)

        assert all(item["일치"] for item in checks), (
            "기준 결과가 재현되지 않았습니다. 배율 평가를 중단합니다."
        )
        _checked = True

    _set_scale(model, RATES[name])
    _status(
        상태="배율 평가 중",
        배율=RATES[name],
        완료=0,
        전체=400,
        결과폴더=str(JOB),
    )

def _progress(name, count):
    if count % 10 == 0:
        _status(
            상태="배율 평가 중",
            배율=RATES[name],
            완료=count,
            전체=400,
            결과폴더=str(JOB),
        )

def _finish():
    report = []

    for name, rate in RATES.items():
        values = json.loads(
            (output / f"{name}_predictions.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(values) == 400

        recovered = regressed = gained = lost = 0

        for index, value in enumerate(values):
            baseline = _cached["base"][index]
            full = _cached["lr_2e-4"][index]

            assert (
                value["번호"]
                == baseline["번호"]
                == full["번호"]
                == index
            )
            assert (
                value["기준 정답"]
                == baseline["기준 정답"]
                == full["기준 정답"]
            )

            correct = bool(value["정답 여부"])
            recovered += int(
                not baseline["정답 여부"] and correct
            )
            regressed += int(
                baseline["정답 여부"] and not correct
            )
            gained += int(
                not full["정답 여부"] and correct
            )
            lost += int(
                full["정답 여부"] and not correct
            )

        report.append({
            "배율": rate,
            "정답 수": sum(
                bool(value["정답 여부"]) for value in values
            ),
            "전체": 400,
            "순정 오답에서 회복": recovered,
            "순정 정답에서 퇴보": regressed,
            "1.0배 오답에서 회복": gained,
            "1.0배 정답에서 퇴보": lost,
        })

    _save(JOB / "scale_comparison.json", {
        "기준 순정 정답": 206,
        "기준 1.0배 정답": 262,
        "결과": report,
    })
    _status(
        상태="세 배율 평가 완료",
        결과=report,
        결과파일=str(JOB / "scale_comparison.json"),
    )

def _probe(model, processor, row, original, index):
    prompt = row['conversations'][0]['value']
    expected_text = row['conversations'][-1]['value']
    expected = extract_answer(expected_text)
    assert expected, f'{index}번 원본 정답 형식이 잘못됐습니다.'
    assert prompt.count('<image>') == 1, f'{index}번 이미지 표기 개수가 잘못됐습니다.'
    before, after = prompt.split('<image>')
    image_path = data / row['image']
    assert image_path.is_file(), f'이미지 파일이 없습니다: {image_path}'
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    content = []
    if before.strip():
        content.append({'type': 'text', 'text': before})
    content.append({'type': 'image', 'image': image})
    if after.strip():
        content.append({'type': 'text', 'text': after})
    messages = [{'role': 'user', 'content': content}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
    input_length = inputs['input_ids'].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1, repetition_penalty=1.0, use_cache=True)
    new_tokens = generated[0, input_length:]
    raw_output = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    predicted = extract_answer(raw_output)
    correct = bool(predicted) and normalize(predicted) == normalize(expected)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    last_token = int(new_tokens[-1]) if len(new_tokens) else None
    exhausted = len(new_tokens) >= 128 and last_token not in eos_ids
    image.close()
    return (predicted, correct)
PREFLIGHT = '\n_cached = {\n    name: json.loads(\n        (REFERENCE / f"{name}_predictions.json").read_text(\n            encoding="utf-8"\n        )\n    )\n    for name in ["base", "lr_2e-4"]\n}\n_provenance = json.loads(\n    (DATA / "internal_dev_provenance.json").read_text(\n        encoding="utf-8"\n    )\n)\n\nfor label, expected in [("base", 206), ("lr_2e-4", 262)]:\n    assert len(_cached[label]) == 400\n    assert sum(\n        bool(value["정답 여부"]) for value in _cached[label]\n    ) == expected\n\n    for index, value in enumerate(_cached[label]):\n        assert value["번호"] == index\n        assert all(\n            value[key] == _provenance[index][key]\n            for key in ["출처", "원본 행 번호", "QA 번호"]\n        )\n\n_hashes = json.loads(\n    (REFERENCE.parent / "data_hashes.json").read_text(\n        encoding="utf-8"\n    )\n)\nassert hashlib.sha256(\n    (DATA / "internal_dev.json").read_bytes()\n).hexdigest() == _hashes["internal_dev.json"]\n\n_save(JOB / "reference_hashes.json", {\n    name: hashlib.sha256(\n        (REFERENCE / f"{name}_predictions.json").read_bytes()\n    ).hexdigest()\n    for name in ["base", "lr_2e-4"]\n})\n'
EVALUATION = "from pathlib import Path\nfrom datetime import datetime\nfrom collections import Counter\nimport gc\nimport json\nimport re\nimport unicodedata\nimport torch\nfrom PIL import Image\nfrom transformers import AutoProcessor, Qwen3VLForConditionalGeneration\nfrom peft import PeftModel\nroot = Path('/shared_dir/mmdl')\nbase = root / 'models/Qwen3-VL-4B-Instruct'\nrun = root / 'checkpoints/expanded_lr_20261116_223345_306888'\ndata = root / 'data/recipe_a_answer_only_20261115_004649_140669'\nrows = json.loads((data / 'internal_dev.json').read_text(encoding='utf-8'))\nprovenance = json.loads((data / 'internal_dev_provenance.json').read_text(encoding='utf-8'))\nassert len(rows) == len(provenance) == 400, '개발 데이터와 원본 기록의 수가 일치하지 않습니다.'\nexperiments = [('scale_110', run / 'lr_2e-4'), ('scale_125', run / 'lr_2e-4'), ('scale_150', run / 'lr_2e-4')]\nfor name, adapter in experiments:\n    if adapter is not None:\n        assert (adapter / 'adapter_model.safetensors').is_file(), f'어댑터 파일이 없습니다: {adapter}'\nstamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')\noutput = JOB / 'results'\noutput.mkdir(parents=True, exist_ok=False)\nprocessor = AutoProcessor.from_pretrained(base, local_files_only=True)\nimage_processor = processor.image_processor\nimage_processor.min_pixels = 3136\nimage_processor.max_pixels = 200704\nif isinstance(getattr(image_processor, 'size', None), dict):\n    image_processor.size['shortest_edge'] = 3136\n    image_processor.size['longest_edge'] = 200704\n\ndef normalize(value):\n    value = unicodedata.normalize('NFKC', value)\n    return ' '.join(value.casefold().split()).strip()\n\ndef extract_answer(value):\n    matches = re.findall('^\\\\s*Answer\\\\s*:\\\\s*(.*?)\\\\s*$', value, flags=re.IGNORECASE | re.MULTILINE)\n    return matches[-1].strip() if matches else ''\nsummaries = []\nfor name, adapter in experiments:\n    print(f'\\n=== {name} 모델 로딩 ===', flush=True)\n    torch.manual_seed(42)\n    torch.cuda.manual_seed_all(42)\n    model = Qwen3VLForConditionalGeneration.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='flash_attention_2')\n    if adapter is not None:\n        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)\n    model = model.to('cuda')\n    model.eval()\n    model.config.use_cache = True\n    _configure(model, name)\n    results = []\n    counts = Counter()\n    source_counts = {}\n    predictions_path = output / f'{name}_predictions.json'\n    for index, (row, original) in enumerate(zip(rows, provenance)):\n        prompt = row['conversations'][0]['value']\n        expected_text = row['conversations'][-1]['value']\n        expected = extract_answer(expected_text)\n        assert expected, f'{index}번 원본 정답 형식이 잘못됐습니다.'\n        assert prompt.count('<image>') == 1, f'{index}번 이미지 표기 개수가 잘못됐습니다.'\n        before, after = prompt.split('<image>')\n        image_path = data / row['image']\n        assert image_path.is_file(), f'이미지 파일이 없습니다: {image_path}'\n        with Image.open(image_path) as opened:\n            image = opened.convert('RGB')\n        content = []\n        if before.strip():\n            content.append({'type': 'text', 'text': before})\n        content.append({'type': 'image', 'image': image})\n        if after.strip():\n            content.append({'type': 'text', 'text': after})\n        messages = [{'role': 'user', 'content': content}]\n        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')\n        input_length = inputs['input_ids'].shape[-1]\n        with torch.inference_mode():\n            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1, repetition_penalty=1.0, use_cache=True)\n        new_tokens = generated[0, input_length:]\n        raw_output = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)\n        predicted = extract_answer(raw_output)\n        correct = bool(predicted) and normalize(predicted) == normalize(expected)\n        eos = model.generation_config.eos_token_id\n        eos_ids = set(eos if isinstance(eos, list) else [eos])\n        last_token = int(new_tokens[-1]) if len(new_tokens) else None\n        exhausted = len(new_tokens) >= 128 and last_token not in eos_ids\n        tag = original['출처']\n        source_counts.setdefault(tag, Counter())\n        counts['문항 수'] += 1\n        counts['정답 수'] += int(correct)\n        counts['답 형식 누락 수'] += int(not predicted)\n        counts['길이 한도 도달 수'] += int(exhausted)\n        source_counts[tag]['문항 수'] += 1\n        source_counts[tag]['정답 수'] += int(correct)\n        results.append({'번호': index, '출처': tag, '이미지': row['image'], '원본 행 번호': original['원본 행 번호'], 'QA 번호': original['QA 번호'], '기준 정답': expected, '추출한 답': predicted, '정답 여부': correct, '생성 토큰 수': len(new_tokens), '길이 한도 도달': exhausted, '전체 출력': raw_output})\n        if (index + 1) % 10 == 0:\n            predictions_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')\n            print(f'{name}: {index + 1}/400개 완료, 현재 정답 {counts['정답 수']}개', flush=True)\n        del inputs, generated, new_tokens\n        image.close()\n        _progress(name, index + 1)\n    summary = {'모델': name, '문항 수': counts['문항 수'], '정답 수': counts['정답 수'], '정확도': counts['정답 수'] / counts['문항 수'], '답 형식 누락 수': counts['답 형식 누락 수'], '길이 한도 도달 수': counts['길이 한도 도달 수'], '출처별 결과': {tag: {'문항 수': values['문항 수'], '정답 수': values['정답 수']} for tag, values in source_counts.items()}}\n    summaries.append(summary)\n    predictions_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')\n    (output / f'{name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')\n    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)\n    del model\n    gc.collect()\n    torch.cuda.empty_cache()\nreport = {'평가 대상': 'Internal Dev 400개', '평가 설정': {'생성 방식': 'greedy', '최대 생성 토큰 수': 128, '최소 이미지 픽셀': 3136, '최대 이미지 픽셀': 200704, '답 추출': '마지막 Answer 줄', '정답 비교': '유니코드·대소문자·공백 정규화 후 일치'}, '결과': summaries}\n(output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')\nprint('\\n=== Internal Dev 생성 평가 완료 ===')\nfor item in summaries:\n    print(f'{item['모델']} | 정답 {item['정답 수']}/{item['문항 수']} | 정확도 {item['정확도']:.1%} | 답 형식 누락 {item['답 형식 누락 수']}개 | 길이 한도 도달 {item['길이 한도 도달 수']}개')\nprint('결과 폴더:', output)"

_lock = (JOB.parent / "lora_scale_eval.lock").open("a")
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (JOB.parent / "latest_lora_scale.txt").write_text(
        str(JOB), encoding="utf-8"
    )
    _status(
        상태="기준 파일 확인 중",
        결과폴더=str(JOB),
    )
    exec(PREFLIGHT, globals())
    exec(EVALUATION, globals())
    _finish()
except Exception:
    _status(
        상태="오류로 중단",
        오류=traceback.format_exc(),
        결과폴더=str(JOB),
    )
    raise
