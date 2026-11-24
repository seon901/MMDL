from pathlib import Path
from datetime import datetime
import ast
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT = Path('/shared_dir/mmdl')
OLD = ROOT / 'checkpoints/replay_compare_20261119_214637_363626'
JOB = ROOT / 'checkpoints/r20_strength_20261120'
REPO = ROOT / 'code/Qwen3-VL/qwen-vl-finetune'
DATA = OLD / 'data/R20'
NEW = JOB / 'lr_1e-4'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def snapshot(path, source):
    compile(source, str(path), 'exec')
    assert all(ord(c) < 128 or 44032 <= ord(c) <= 55203 for c in source), '코드 문자 검사 실패'
    if path.exists():
        assert path.read_text(encoding='utf-8') == source, f'저장된 코드가 다릅니다: {path}'
    else:
        path.write_text(source, encoding='utf-8')

def patch_globals(tree, values):
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in values:
                    node.value = ast.parse(values[target.id], mode='eval').body
                    found.add(target.id)
    assert found == set(values), f'코드 구조 확인 실패: {found}'
    return tree

def run_script(script, log):
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-u', str(script)], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

def image_code(label, adapter, output):
    tree = ast.parse((OLD / 'evaluate_R20.py').read_text(encoding='utf-8'))
    patch_globals(tree, {
        'experiments': repr([(label, str(adapter))]),
        'output': f'Path({str(output)!r})',
    })
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'experiments' for t in node.targets):
            pair = node.value.elts[0]
            pair.elts[1] = ast.parse(f'Path({str(adapter)!r})', mode='eval').body
    return ast.unparse(ast.fix_missing_locations(tree)).replace(chr(183), ' ') + '\n'

def text_code(adapters):
    tree = ast.parse((OLD / 'evaluate_text.py').read_text(encoding='utf-8'))
    patch_globals(tree, {'JOB': f'Path({str(JOB)!r})'})
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cutoff = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'results' for t in n.targets))
    main.body = main.body[:cutoff]
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'label')
    loop.iter = ast.parse(repr(list(adapters)), mode='eval').body
    folder = loop.body[0]
    assert isinstance(folder, ast.Assign) and folder.targets[0].id == 'folder'
    folder.value = ast.parse(f'Path({repr({k: str(v) for k, v in adapters.items()})}[label])', mode='eval').body
    assert isinstance(loop.body[1], ast.Assert)
    loop.body.pop(1)
    main.body.extend(ast.parse("save(JOB / 'text_all_predictions.json', predictions)").body)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))]
    tree.body.extend(ast.parse('main()').body)
    return ast.unparse(ast.fix_missing_locations(tree)).replace(chr(183), ' ') + '\n'

def main():
    assert hashlib.sha256((DATA / 'train.json').read_bytes()).hexdigest() == 'ef0fa3a5011e5acc02479bc2dce7a7d16129ae316514f9b6b6f3392c4474ef1b', 'R20 학습 데이터가 변경됐습니다.'
    assert hashlib.sha256((DATA / 'internal_dev.json').read_bytes()).hexdigest() == '90d8b79f0fe3c233dc91bb0738d3a367ced1e864214434c9608b76bc285ad7fb', '개발 데이터가 변경됐습니다.'
    for filename in ['entry.py', 'evaluate_R20.py', 'evaluate_text.py', 'R20/training_arguments.txt']:
        assert (OLD / filename).is_file(), f'필요한 파일이 없습니다: {OLD / filename}'
    os.environ.update(PILOT_REPO=str(REPO), PILOT_DATA=str(DATA), CHECK_DATA='1',
                      CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    NEW.mkdir(exist_ok=True)
    snapshot(JOB / 'entry.py', (OLD / 'entry.py').read_text(encoding='utf-8'))
    arguments = [line for line in (OLD / 'R20/training_arguments.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    for key, value in {'--output_dir': str(NEW), '--learning_rate': '1e-4', '--run_name': 'r20_strength_lr_1e-4'}.items():
        assert key in arguments, f'필요한 학습 옵션이 없습니다: {key}'
        arguments[arguments.index(key) + 1] = value
    assert arguments[arguments.index('--max_steps') + 1] == '500'
    assert arguments[arguments.index('--lora_r') + 1] == '8'
    argument_path = NEW / 'training_arguments.txt'
    argument_text = '\n'.join(arguments) + '\n'
    if argument_path.exists():
        assert argument_path.read_text(encoding='utf-8') == argument_text
    else:
        argument_path.write_text(argument_text, encoding='utf-8')
    save(JOB / 'experiment_plan.json', {
        '목적': '기존 R20의 학습률과 중간 체크포인트 비교',
        '기존 R20': str(OLD / 'R20'), '새 학습률': 0.0001,
        '학습 문항 수': 4000, '학습 단계': 500,
        '고정 조건': '기존 R20 데이터, 지시문, 시드, rank, alpha, 전처리와 생성 평가',
        '시작 모델': '순정 모델. 중단된 새 실험만 자체 체크포인트에서 재개.',
        '선택 원칙': '이미지와 텍스트 개발 점수 및 순정 정답의 퇴보를 함께 비교. MMMU 결과를 선택에 사용하지 않음.',
    })
    save(JOB / 'source_hashes.json', {name: hashlib.sha256((OLD / name).read_bytes()).hexdigest() for name in ['entry.py', 'evaluate_R20.py', 'evaluate_text.py', 'R20/training_arguments.txt']})
    done = (NEW / 'pilot_result.json').is_file() and (NEW / 'adapter_model.safetensors').is_file() and read(NEW / 'pilot_result.json')['완료 단계'] == 500
    if not done:
        status('학습률 1e-4 학습 중', 로그=str(NEW / 'training.log'))
        with (NEW / 'training.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1', str(JOB / 'entry.py'), *arguments], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(NEW / 'trainer_state.json')['global_step'] == 500
    adapters = {'reference_R20': OLD / 'R20', 'lr1e4_step500': NEW}
    skipped = []
    for label, parent in [('lr2e4_step300', OLD / 'R20'), ('lr1e4_step300', NEW)]:
        checkpoint = parent / 'checkpoint-300'
        if (checkpoint / 'adapter_model.safetensors').is_file() and (checkpoint / 'adapter_config.json').is_file() and (checkpoint / 'trainer_state.json').is_file():
            assert read(checkpoint / 'trainer_state.json')['global_step'] == 300
            adapters[label] = checkpoint
        else:
            skipped.append({'구성': label, '사유': '300단계 체크포인트가 남아 있지 않음'})
    save(JOB / 'evaluation_plan.json', {'모델': {k: str(v) for k, v in adapters.items()}, '생략': skipped})
    image_predictions = {'reference_R20': read(OLD / 'evaluation_R20/R20_predictions.json')}
    for label, adapter in adapters.items():
        if label == 'reference_R20':
            continue
        output = JOB / ('image_' + label)
        script = JOB / ('evaluate_' + label + '.py')
        snapshot(script, image_code(label, adapter, output))
        prediction_file = output / (label + '_predictions.json')
        complete = prediction_file.is_file() and len(read(prediction_file)) == 400 and (output / 'comparison.json').is_file()
        if not complete:
            status('이미지 개발 평가 중', 구성=label, 로그=str(JOB / (label + '_evaluation.log')))
            run_script(script, JOB / (label + '_evaluation.log'))
        image_predictions[label] = read(prediction_file)
    cached = JOB / 'text_reference_R20_predictions.json'
    if not cached.exists():
        shutil.copy2(OLD / 'text_R20_predictions.json', cached)
    text_script = JOB / 'evaluate_text.py'
    snapshot(text_script, text_code(adapters))
    status('텍스트 개발 평가 중', 로그=str(JOB / 'text_evaluation.log'))
    run_script(text_script, JOB / 'text_evaluation.log')
    text_predictions = read(JOB / 'text_all_predictions.json')
    image_base = read(ROOT / 'checkpoints/expanded_lr_20261116_223345_306888/generation_eval_20261116_231128_946555/base_predictions.json')
    report = []
    for label, values in image_predictions.items():
        assert len(values) == len(image_base) == 400
        assert all(all(row[k] == base[k] for k in ['번호', '출처', '원본 행 번호', 'QA 번호', '기준 정답']) for row, base in zip(values, image_base))
        texts = text_predictions[label]
        text_base = text_predictions['base']
        assert len(texts) == len(text_base) == 300
        assert all(row['input_sha256'] == base['input_sha256'] for row, base in zip(texts, text_base))
        report.append({
            '구성': label,
            '이미지 정답 수': sum(bool(r['정답 여부']) for r in values),
            '이미지 순정 정답 퇴보': sum(bool(b['정답 여부']) and not r['정답 여부'] for b, r in zip(image_base, values)),
            '텍스트 정답 수': sum(bool(r['correct']) for r in texts),
            '텍스트 순정 정답 퇴보': sum(bool(b['correct']) and not r['correct'] for b, r in zip(text_base, texts)),
            '이미지 길이 한도 도달': sum(bool(r['길이 한도 도달']) for r in values),
            '텍스트 길이 한도 도달': sum(bool(r['exhausted']) for r in texts),
        })
    assert report[0]['이미지 정답 수'] == 259 and report[0]['텍스트 정답 수'] == 259
    save(JOB / 'strength_comparison.json', {'결과': report, '생략': skipped, '주의': '단일 시드의 내부 개발 탐색 결과. MMMU와 MMMU-Pro는 실행하지 않았습니다.'})
    status('학습과 이미지 및 텍스트 평가 완료', 결과=report, 결과파일=str(JOB / 'strength_comparison.json'))
    print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 학교 GPU 비교 작업이 실행 중입니다. 중복 실행을 중단합니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
