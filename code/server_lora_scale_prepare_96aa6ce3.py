from pathlib import Path
from datetime import datetime
import ast
import copy
import hashlib
import json
import subprocess
import sys
ROOT = Path('/shared_dir/mmdl')
RUN = ROOT / 'checkpoints/expanded_lr_20261116_223345_306888'
SOURCE = RUN / 'evaluate_expanded_dev.py'
REFERENCE = RUN / 'generation_eval_20261116_231128_946555'
RATES = {'scale_110': 1.1, 'scale_125': 1.25, 'scale_150': 1.5}
HELPERS = '\ndef _save(path, value):\n    temporary = path.with_suffix(path.suffix + ".tmp")\n    temporary.write_text(\n        json.dumps(value, ensure_ascii=True, indent=2),\n        encoding="utf-8",\n    )\n    temporary.replace(path)\n\ndef _status(**value):\n    _save(JOB / "status.json", value)\n\ndef _set_scale(model, value):\n    from peft.tuners.lora.layer import LoraLayer\n\n    active = model.active_adapters\n    assert len(active) == 1\n    adapter = active[0]\n    assert not getattr(\n        model.peft_config[adapter], "use_rslora", False\n    )\n\n    count = 0\n    for module_name, layer in model.named_modules():\n        if isinstance(layer, LoraLayer) and adapter in layer.scaling:\n            assert "language_model" in module_name\n            assert "visual" not in module_name\n\n            layer.set_scale(adapter, value)\n            expected = (\n                value * layer.lora_alpha[adapter] / layer.r[adapter]\n            )\n            assert abs(layer.scaling[adapter] - expected) < 1e-9\n            count += 1\n\n    assert count == 144, (\n        f"LoRA 대상 모듈 수가 다릅니다: {count}"\n    )\n\ndef _configure(model, name):\n    global _checked\n\n    if not _checked:\n        _status(\n            상태="기준 결과 재현 확인 중",\n            결과폴더=str(JOB),\n        )\n\n        groups = {}\n        for index, original in enumerate(provenance):\n            groups.setdefault(original["출처"], []).append(index)\n\n        indices = [\n            index\n            for tag in sorted(groups)\n            for index in groups[tag][:2]\n        ]\n        assert len(indices) == 8\n\n        checks = []\n        _set_scale(model, 1.0)\n\n        for label in ["base", "lr_2e-4"]:\n            for index in indices:\n                if label == "base":\n                    with model.disable_adapter():\n                        prediction, correct = _probe(\n                            model,\n                            processor,\n                            rows[index],\n                            provenance[index],\n                            index,\n                        )\n                else:\n                    prediction, correct = _probe(\n                        model,\n                        processor,\n                        rows[index],\n                        provenance[index],\n                        index,\n                    )\n\n                cached = _cached[label][index]\n                matches = (\n                    normalize(prediction)\n                    == normalize(cached["추출한 답"])\n                    and correct == cached["정답 여부"]\n                )\n                checks.append({\n                    "모델": label,\n                    "번호": index,\n                    "일치": matches,\n                    "기존 답": cached["추출한 답"],\n                    "현재 답": prediction,\n                })\n\n        _save(JOB / "reference_check.json", checks)\n\n        assert all(item["일치"] for item in checks), (\n            "기준 결과가 재현되지 않았습니다. 배율 평가를 중단합니다."\n        )\n        _checked = True\n\n    _set_scale(model, RATES[name])\n    _status(\n        상태="배율 평가 중",\n        배율=RATES[name],\n        완료=0,\n        전체=400,\n        결과폴더=str(JOB),\n    )\n\ndef _progress(name, count):\n    if count % 10 == 0:\n        _status(\n            상태="배율 평가 중",\n            배율=RATES[name],\n            완료=count,\n            전체=400,\n            결과폴더=str(JOB),\n        )\n\ndef _finish():\n    report = []\n\n    for name, rate in RATES.items():\n        values = json.loads(\n            (output / f"{name}_predictions.json").read_text(\n                encoding="utf-8"\n            )\n        )\n        assert len(values) == 400\n\n        recovered = regressed = gained = lost = 0\n\n        for index, value in enumerate(values):\n            baseline = _cached["base"][index]\n            full = _cached["lr_2e-4"][index]\n\n            assert (\n                value["번호"]\n                == baseline["번호"]\n                == full["번호"]\n                == index\n            )\n            assert (\n                value["기준 정답"]\n                == baseline["기준 정답"]\n                == full["기준 정답"]\n            )\n\n            correct = bool(value["정답 여부"])\n            recovered += int(\n                not baseline["정답 여부"] and correct\n            )\n            regressed += int(\n                baseline["정답 여부"] and not correct\n            )\n            gained += int(\n                not full["정답 여부"] and correct\n            )\n            lost += int(\n                full["정답 여부"] and not correct\n            )\n\n        report.append({\n            "배율": rate,\n            "정답 수": sum(\n                bool(value["정답 여부"]) for value in values\n            ),\n            "전체": 400,\n            "순정 오답에서 회복": recovered,\n            "순정 정답에서 퇴보": regressed,\n            "1.0배 오답에서 회복": gained,\n            "1.0배 정답에서 퇴보": lost,\n        })\n\n    _save(JOB / "scale_comparison.json", {\n        "기준 순정 정답": 206,\n        "기준 1.0배 정답": 262,\n        "결과": report,\n    })\n    _status(\n        상태="세 배율 평가 완료",\n        결과=report,\n        결과파일=str(JOB / "scale_comparison.json"),\n    )\n'

def build(source):
    tree = ast.parse(source)
    experiments = next((node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'experiments' for target in node.targets))))
    expected = ast.parse('[("base", None), ("lr_1e-4", run / "lr_1e-4"), ("lr_2e-4", run / "lr_2e-4")]', mode='eval').body
    assert ast.dump(experiments.value) == ast.dump(expected)
    experiments.value = ast.parse('[("scale_110", run / "lr_2e-4"), ("scale_125", run / "lr_2e-4"), ("scale_150", run / "lr_2e-4")]', mode='eval').body
    target = next((node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(item, ast.Name) and item.id == 'output' for item in node.targets))))
    target.value = ast.parse('JOB / "results"', mode='eval').body

    def has_cache_assignment(node):
        return isinstance(node, ast.Assign) and any((isinstance(target, ast.Attribute) and target.attr == 'use_cache' for target in node.targets))
    model_loop = next((node for node in tree.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and (node.iter.id == 'experiments') and any((has_cache_assignment(item) for item in node.body))))
    position = next((index for index, node in enumerate(model_loop.body) if has_cache_assignment(node)))
    model_loop.body.insert(position + 1, ast.parse('_configure(model, name)').body[0])
    sample_loop = next((node for node in model_loop.body if isinstance(node, ast.For)))
    stop = next((index for index, node in enumerate(sample_loop.body) if isinstance(node, ast.Assign) and any((isinstance(item, ast.Name) and item.id == 'tag' for item in node.targets))))
    probe = ast.parse('def _probe(model, processor, row, original, index):\n    pass').body[0]
    probe.body = copy.deepcopy(sample_loop.body[:stop]) + ast.parse('image.close()\nreturn predicted, correct').body
    sample_loop.body.append(ast.parse('_progress(name, index + 1)').body[0])
    ast.fix_missing_locations(tree)
    ast.fix_missing_locations(probe)
    return (ast.unparse(tree), ast.unparse(probe))

def main():
    source = SOURCE.read_text(encoding='utf-8')
    evaluation, probe = build(source)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    job = ROOT / f'checkpoints/lora_scale_{stamp}'
    job.mkdir(exist_ok=False)
    prologue = f'from pathlib import Path\nimport fcntl, hashlib, json, traceback\nJOB = Path(__file__).resolve().parent\nRATES = {RATES!r}\nREFERENCE = Path({str(REFERENCE)!r})\nDATA = Path("/shared_dir/mmdl/data/recipe_a_answer_only_20261115_004649_140669")\n_checked = False\n'
    preflight = '\n_cached = {\n    name: json.loads(\n        (REFERENCE / f"{name}_predictions.json").read_text(\n            encoding="utf-8"\n        )\n    )\n    for name in ["base", "lr_2e-4"]\n}\n_provenance = json.loads(\n    (DATA / "internal_dev_provenance.json").read_text(\n        encoding="utf-8"\n    )\n)\n\nfor label, expected in [("base", 206), ("lr_2e-4", 262)]:\n    assert len(_cached[label]) == 400\n    assert sum(\n        bool(value["정답 여부"]) for value in _cached[label]\n    ) == expected\n\n    for index, value in enumerate(_cached[label]):\n        assert value["번호"] == index\n        assert all(\n            value[key] == _provenance[index][key]\n            for key in ["출처", "원본 행 번호", "QA 번호"]\n        )\n\n_hashes = json.loads(\n    (REFERENCE.parent / "data_hashes.json").read_text(\n        encoding="utf-8"\n    )\n)\nassert hashlib.sha256(\n    (DATA / "internal_dev.json").read_bytes()\n).hexdigest() == _hashes["internal_dev.json"]\n\n_save(JOB / "reference_hashes.json", {\n    name: hashlib.sha256(\n        (REFERENCE / f"{name}_predictions.json").read_bytes()\n    ).hexdigest()\n    for name in ["base", "lr_2e-4"]\n})\n'
    tail = '\n_lock = (JOB.parent / "lora_scale_eval.lock").open("a")\ntry:\n    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n    (JOB.parent / "latest_lora_scale.txt").write_text(\n        str(JOB), encoding="utf-8"\n    )\n    _status(\n        상태="기준 파일 확인 중",\n        결과폴더=str(JOB),\n    )\n    exec(PREFLIGHT, globals())\n    exec(EVALUATION, globals())\n    _finish()\nexcept Exception:\n    _status(\n        상태="오류로 중단",\n        오류=traceback.format_exc(),\n        결과폴더=str(JOB),\n    )\n    raise\n'
    worker = prologue + HELPERS + '\n' + probe + '\nPREFLIGHT = ' + repr(preflight) + '\nEVALUATION = ' + repr(evaluation) + '\n' + tail
    ast.parse(worker)
    assert all((ord(char) < 128 or 44032 <= ord(char) <= 55203 or ord(char) == 183 for char in worker)), '문자 검사에 실패했습니다. 평가를 시작하지 않습니다.'
    (job / 'evaluate.py').write_text(worker, encoding='utf-8')
    (job / 'prepare.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    (job / 'source_evaluator.py').write_text(source, encoding='utf-8')
    (job / 'manifest.json').write_text(json.dumps({'scales': RATES, 'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(), 'reference_directory': str(REFERENCE), 'purpose': 'Internal development comparison; not MMMU evaluation'}, indent=2), encoding='utf-8')
    with (job / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(job / 'evaluate.py')], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('배율 비교 실행 요청 완료')
    print('결과 폴더:', job)
    print('프로세스 번호:', process.pid)
if __name__ == '__main__':
    main()
