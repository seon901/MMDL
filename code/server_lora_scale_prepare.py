from pathlib import Path
from datetime import datetime
import ast
import copy
import hashlib
import json
import subprocess
import sys

ROOT = Path("/shared_dir/mmdl")
RUN = ROOT / "checkpoints/expanded_lr_20261116_223345_306888"
SOURCE = RUN / "evaluate_expanded_dev.py"
REFERENCE = RUN / "generation_eval_20261116_231128_946555"
RATES = {
    "scale_060": 0.6,
    "scale_080": 0.8,
    "scale_090": 0.9,
}

HELPERS = r'''
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
'''

def build(source):
    tree = ast.parse(source)

    experiments = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "experiments"
            for target in node.targets
        )
    )

    expected = ast.parse(
        '[("base", None), '
        '("lr_1e-4", run / "lr_1e-4"), '
        '("lr_2e-4", run / "lr_2e-4")]',
        mode="eval",
    ).body
    assert ast.dump(experiments.value) == ast.dump(expected)

    experiments.value = ast.parse(
        '[("scale_060", run / "lr_2e-4"), '
        '("scale_080", run / "lr_2e-4"), '
        '("scale_090", run / "lr_2e-4")]',
        mode="eval",
    ).body

    target = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(item, ast.Name)
            and item.id == "output"
            for item in node.targets
        )
    )
    target.value = ast.parse(
        'JOB / "results"', mode="eval"
    ).body

    def has_cache_assignment(node):
        return (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "use_cache"
                for target in node.targets
            )
        )

    model_loop = next(
        node for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "experiments"
        and any(has_cache_assignment(item) for item in node.body)
    )

    position = next(
        index for index, node in enumerate(model_loop.body)
        if has_cache_assignment(node)
    )
    model_loop.body.insert(
        position + 1,
        ast.parse("_configure(model, name)").body[0],
    )

    sample_loop = next(
        node for node in model_loop.body
        if isinstance(node, ast.For)
    )
    stop = next(
        index for index, node in enumerate(sample_loop.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(item, ast.Name)
            and item.id == "tag"
            for item in node.targets
        )
    )

    probe = ast.parse(
        "def _probe(model, processor, row, original, index):\n"
        "    pass"
    ).body[0]
    probe.body = (
        copy.deepcopy(sample_loop.body[:stop])
        + ast.parse(
            "image.close()\nreturn predicted, correct"
        ).body
    )

    sample_loop.body.append(
        ast.parse("_progress(name, index + 1)").body[0]
    )
    ast.fix_missing_locations(tree)
    ast.fix_missing_locations(probe)
    return ast.unparse(tree), ast.unparse(probe)

def main():
    source = SOURCE.read_text(encoding="utf-8")
    evaluation, probe = build(source)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job = ROOT / f"checkpoints/lora_scale_{stamp}"
    job.mkdir(exist_ok=False)

    prologue = f'''from pathlib import Path
import fcntl, hashlib, json, traceback
JOB = Path(__file__).resolve().parent
RATES = {RATES!r}
REFERENCE = Path({str(REFERENCE)!r})
DATA = Path("/shared_dir/mmdl/data/recipe_a_answer_only_20261115_004649_140669")
_checked = False
'''

    preflight = '''
_cached = {
    name: json.loads(
        (REFERENCE / f"{name}_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    for name in ["base", "lr_2e-4"]
}
_provenance = json.loads(
    (DATA / "internal_dev_provenance.json").read_text(
        encoding="utf-8"
    )
)

for label, expected in [("base", 206), ("lr_2e-4", 262)]:
    assert len(_cached[label]) == 400
    assert sum(
        bool(value["정답 여부"]) for value in _cached[label]
    ) == expected

    for index, value in enumerate(_cached[label]):
        assert value["번호"] == index
        assert all(
            value[key] == _provenance[index][key]
            for key in ["출처", "원본 행 번호", "QA 번호"]
        )

_hashes = json.loads(
    (REFERENCE.parent / "data_hashes.json").read_text(
        encoding="utf-8"
    )
)
assert hashlib.sha256(
    (DATA / "internal_dev.json").read_bytes()
).hexdigest() == _hashes["internal_dev.json"]

_save(JOB / "reference_hashes.json", {
    name: hashlib.sha256(
        (REFERENCE / f"{name}_predictions.json").read_bytes()
    ).hexdigest()
    for name in ["base", "lr_2e-4"]
})
'''

    tail = '''
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
'''

    worker = (
        prologue
        + HELPERS
        + "\n"
        + probe
        + "\nPREFLIGHT = "
        + repr(preflight)
        + "\nEVALUATION = "
        + repr(evaluation)
        + "\n"
        + tail
    )
    ast.parse(worker)

    assert all(
        ord(char) < 128
        or 0xAC00 <= ord(char) <= 0xD7A3
        or ord(char) == 183
        for char in worker
    ), "문자 검사에 실패했습니다. 평가를 시작하지 않습니다."

    (job / "evaluate.py").write_text(worker, encoding="utf-8")
    (job / "prepare.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (job / "source_evaluator.py").write_text(
        source, encoding="utf-8"
    )
    (job / "manifest.json").write_text(
        json.dumps({
            "scales": RATES,
            "source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "reference_directory": str(REFERENCE),
            "purpose": (
                "Internal development comparison; not MMMU evaluation"
            ),
        }, indent=2),
        encoding="utf-8",
    )

    with (job / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(job / "evaluate.py")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print("배율 비교 실행 요청 완료")
    print("결과 폴더:", job)
    print("프로세스 번호:", process.pid)

if __name__ == "__main__":
    main()
