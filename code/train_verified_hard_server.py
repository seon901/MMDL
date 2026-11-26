from pathlib import Path
from datetime import datetime
from collections import Counter
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback

ROOT = Path('/shared_dir/mmdl')
OLD = ROOT / 'checkpoints/replay_compare_20261119_214637_363626'
PROFILE = ROOT / 'checkpoints/verified_hard_profile_20261121'
POOL = ROOT / 'data/engineering83_prepared_20261118/paired_candidate_split_20261119_085350_215640'
GROUPS = ROOT / 'data/engineering83_prepared_20261118/validation_20261118_230546_267103/contamination_20261118_231821_678269/split_groups.json'
JOB = ROOT / 'checkpoints/verified_hard_compare_20261121'
REPO = ROOT / 'code/Qwen3-VL/qwen-vl-finetune'
ARMS = ['random_answer', 'hard_answer']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def fixed_json(path, value):
    if path.exists():
        assert read(path) == value, f'기존 파일과 다릅니다: {path}'
    else:
        save(path, value)

def load_helper():
    path = ROOT / 'r20_strength_server.py'
    spec = importlib.util.spec_from_file_location('strength_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.JOB = JOB
    return module

def absolute_row(row, folder):
    result = copy.deepcopy(row)
    if result.get('image'):
        assert isinstance(result['image'], str)
        image = Path(result['image'])
        image = image if image.is_absolute() else folder / image
        image = image.resolve()
        assert image.is_file(), f'이미지가 없습니다: {image}'
        result['image'] = str(image)
    return result

def prepare():
    profile = read(PROFILE / 'predictions.json')
    engineering = read(POOL / 'answer_only_train.json')
    engineering_dev = read(POOL / 'answer_only_internal_dev.json')
    assert len(profile) == len(engineering) == 63 and len(engineering_dev) == 16
    pool = {row['id']: row for row in engineering}
    assert len(pool) == 63 and {v['id'] for v in profile} == set(pool)
    dev_ids = {row['id'] for row in engineering_dev}
    mapping = {candidate: group['group_id'] for group in read(GROUPS) for candidate in group['members']}
    assert not set(pool) & dev_ids
    assert not {mapping[k] for k in pool} & {mapping[k] for k in dev_ids}
    plan = read(PROFILE / 'profile_plan.json')
    assert plan['학습 JSON SHA256'] == sha(POOL / 'answer_only_train.json')
    assert plan['어댑터 SHA256'] == sha(OLD / 'R20/adapter_model.safetensors')
    for item in profile:
        assert item['group'] == mapping[item['id']]
    eligible = [item for item in profile if not item['length_limit']]
    hard = []
    used = set()
    for item in sorted(eligible, key=lambda x: (-x['gold_loss'], x['id'])):
        if item['exact_normalized_match'] or item['group'] in used:
            continue
        hard.append(item)
        used.add(item['group'])
    count = min(32, len(hard))
    assert count >= 8, '서로 다른 그룹의 어려운 후보가 너무 적어 자동 학습을 시작하지 않습니다.'
    hard = hard[:count]
    shuffled = sorted(eligible, key=lambda item: item['id'])
    random.Random(42).shuffle(shuffled)
    control = []
    used = set()
    for item in shuffled:
        if item['group'] not in used:
            control.append(item)
            used.add(item['group'])
        if len(control) == count:
            break
    assert len(control) == count
    selected = {'random_answer': control, 'hard_answer': hard}
    original = OLD / 'data/R20'
    assert sha(original / 'train.json') == 'ef0fa3a5011e5acc02479bc2dce7a7d16129ae316514f9b6b6f3392c4474ef1b'
    assert sha(original / 'internal_dev.json') == '90d8b79f0fe3c233dc91bb0738d3a367ced1e864214434c9608b76bc285ad7fb'
    baseline = [absolute_row(row, original) for row in read(original / 'train.json')]
    dev = [absolute_row(row, original) for row in read(original / 'internal_dev.json')]
    assert len(baseline) == 4000 and len(dev) == 400
    image_indices = [i for i, row in enumerate(baseline) if row.get('image')]
    assert len(image_indices) == 3200
    repetitions = 4
    replacements = count * repetitions
    slots = sorted(random.Random(42).sample(image_indices, replacements))
    bindings = []
    for arm in ARMS:
        folder = JOB / 'data' / arm
        folder.mkdir(parents=True, exist_ok=True)
        rows = copy.deepcopy(baseline)
        additions = [absolute_row(pool[item['id']], POOL) for item in selected[arm]]
        expanded = [copy.deepcopy(row) for repeat in range(repetitions) for row in additions]
        for index, replacement in zip(slots, expanded):
            rows[index] = replacement
        assert len(rows) == 4000 and sum(bool(row.get('image')) for row in rows) == 3200
        assert all(rows[i] == baseline[i] for i in range(4000) if i not in slots)
        fixed_json(folder / 'train.json', rows)
        fixed_json(folder / 'internal_dev.json', dev)
        fixed_json(folder / 'selection.json', selected[arm])
        bindings.append({'구성': arm, '학습 SHA256': sha(folder / 'train.json'), '개발 SHA256': sha(folder / 'internal_dev.json'),
                         '추가 고유 문항': count, '반복 수': repetitions, '교체 문항 수': replacements,
                         '선정 문항의 평균 정답 손실': sum(item['gold_loss'] for item in selected[arm]) / count})
    design = {'목적': '검토된 공학 문항의 정답형 보강과 난도 선별 효과 비교', '구성': bindings,
              '학습 수': 4000, '이미지 문항 수': 3200, '텍스트 Replay 수': 800,
              '공통 교체 위치': slots, '두 조건의 공통 추가 문항 수': len({x['id'] for x in hard} & {x['id'] for x in control}),
              '시드': 42, '학습률': 0.0002, 'rank': 8, 'alpha': 16, '학습 단계': 500,
              '시작 모델': '각 조건 모두 순정 모델에서 시작. 기존 R20에 이어 학습하지 않음.',
              '풀이 적용': False, '어려운 후보의 의미': '출력 한도 도달과 표기 일치를 제외하고 정답 손실이 높은 문항. 수식 동치를 전부 판정한 확정 오답 집합은 아님.',
              '주의': '소규모 단일 시드 탐색. 공학 문항 반복 노출을 두 조건에 동일하게 적용. 전체 분야의 최적 구성을 주장하지 않음.'}
    fixed_json(JOB / 'experiment_plan.json', design)
    return design

def engineering_code(label, adapter):
    source = (ROOT / 'mine_verified_hard_server.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    processor = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'processor' for t in n.targets))
    model = ast.parse('model = None').body[0]
    loop = copy.deepcopy(next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)))
    class Counts(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 63:
                return ast.copy_location(ast.Constant(16), node)
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace('/63 완료', '/16 완료')), node)
            return node
    loop = Counts().visit(loop)
    output = JOB / ('engineering_' + label + '.json')
    header = f'''import sys
sys.path.insert(0, {str(ROOT)!r})
from mine_verified_hard_server import *
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
JOB = Path({str(JOB)!r})
def status(state, **extra):
    save(JOB / 'status.json', {{'상태': state, '갱신 시각': datetime.now().isoformat(), **extra}})
ADAPTER = Path({str(adapter)!r})
train = read(DATA / 'answer_only_internal_dev.json')
assert len(train) == 16
mapping = {{candidate: group['group_id'] for group in read(GROUPS) for candidate in group['members']}}
saved = Path({str(output)!r})
values = read(saved) if saved.exists() else []
assert len(values) <= 16
'''
    body = ast.unparse(ast.fix_missing_locations(ast.Module(body=[processor, model, loop], type_ignores=[])))
    return header + body + '\n'

def execute(script, log):
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-u', str(script)], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)

def main():
    helper = load_helper()
    status('정답형 보강 데이터 준비 중')
    design = prepare()
    helper.snapshot(JOB / 'entry.py', (OLD / 'entry.py').read_text(encoding='utf-8'))
    os.environ.update(PILOT_REPO=str(REPO), CHECK_DATA='1', CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    adapters = {'reference_R20': OLD / 'R20'}
    image_predictions = {'reference_R20': read(OLD / 'evaluation_R20/R20_predictions.json')}
    for arm in ARMS:
        folder = JOB / arm
        folder.mkdir(exist_ok=True)
        os.environ['PILOT_DATA'] = str(JOB / 'data' / arm)
        arguments = [line for line in (OLD / 'R20/training_arguments.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        for key, value in {'--output_dir': str(folder), '--run_name': arm}.items():
            assert key in arguments
            arguments[arguments.index(key) + 1] = value
        for key, expected in {'--learning_rate': '2e-4', '--max_steps': '500', '--lora_r': '8', '--lora_alpha': '16', '--seed': '42', '--data_seed': '42'}.items():
            assert arguments[arguments.index(key) + 1] == expected
        text = '\n'.join(arguments) + '\n'
        argument_path = folder / 'training_arguments.txt'
        if argument_path.exists():
            assert argument_path.read_text(encoding='utf-8') == text
        else:
            argument_path.write_text(text, encoding='utf-8')
        done = (folder / 'pilot_result.json').is_file() and (folder / 'adapter_model.safetensors').is_file() and read(folder / 'pilot_result.json')['완료 단계'] == 500
        if not done:
            status('정답형 보강 학습 중', 구성=arm, 로그=str(folder / 'training.log'))
            with (folder / 'training.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1', str(JOB / 'entry.py'), *arguments], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(folder / 'trainer_state.json')['global_step'] == 500
        adapters[arm] = folder
        output = JOB / ('image_' + arm)
        evaluator = JOB / ('evaluate_' + arm + '.py')
        helper.snapshot(evaluator, helper.image_code(arm, folder, output))
        path = output / (arm + '_predictions.json')
        if not (path.exists() and len(read(path)) == 400 and (output / 'comparison.json').is_file()):
            status('이미지 개발 평가 중', 구성=arm, 로그=str(folder / 'evaluation.log'))
            execute(evaluator, folder / 'evaluation.log')
        image_predictions[arm] = read(path)
    cache = JOB / 'text_reference_R20_predictions.json'
    fixed_json(cache, read(OLD / 'text_R20_predictions.json'))
    text_script = JOB / 'evaluate_text.py'
    helper.snapshot(text_script, helper.text_code(adapters))
    status('텍스트 개발 평가 중', 로그=str(JOB / 'text_evaluation.log'))
    execute(text_script, JOB / 'text_evaluation.log')
    texts = read(JOB / 'text_all_predictions.json')
    engineering = {}
    for label, adapter in adapters.items():
        script = JOB / ('evaluate_engineering_' + label + '.py')
        helper.snapshot(script, engineering_code(label, adapter))
        status('공학 개발 16개 평가 중', 구성=label, 로그=str(JOB / ('engineering_' + label + '.log')))
        execute(script, JOB / ('engineering_' + label + '.log'))
        engineering[label] = read(JOB / ('engineering_' + label + '.json'))
    base_image = read(ROOT / 'checkpoints/expanded_lr_20261116_223345_306888/generation_eval_20261116_231128_946555/base_predictions.json')
    report = []
    for label in adapters:
        images, text_values, eng = image_predictions[label], texts[label], engineering[label]
        assert len(images) == 400 and len(text_values) == 300 and len(eng) == 16
        for row, base in zip(images, base_image):
            assert all(row[k] == base[k] for k in ['번호', '출처', '원본 행 번호', 'QA 번호', '기준 정답'])
        for row, base in zip(text_values, texts['base']):
            assert row['input_sha256'] == base['input_sha256']
        for row, reference in zip(eng, engineering['reference_R20']):
            assert row['id'] == reference['id'] and row['input_sha256'] == reference['input_sha256']
        report.append({'구성': label, '이미지 정답 수': sum(bool(r['정답 여부']) for r in images),
                       '이미지 순정 정답 퇴보': sum(bool(b['정답 여부']) and not r['정답 여부'] for b,r in zip(base_image,images)),
                       '텍스트 정답 수': sum(bool(r['correct']) for r in text_values),
                       '텍스트 순정 정답 퇴보': sum(bool(b['correct']) and not r['correct'] for b,r in zip(texts['base'],text_values)),
                       '공학 개발 평균 정답 손실': sum(r['gold_loss'] for r in eng)/16,
                       '공학 개발 정답 표기 일치': sum(r['exact_normalized_match'] for r in eng),
                       '공학 개발 길이 한도 도달': sum(r['length_limit'] for r in eng)})
    assert report[0]['이미지 정답 수'] == 259 and report[0]['텍스트 정답 수'] == 259
    save(JOB / 'hard_comparison.json', {'결과': report, '설계': design, '주의': '공학 표기 일치는 의미적 정확도가 아닙니다. 정답 손실 감소만으로 성능 향상을 확정하지 않습니다. MMMU 평가는 실행하지 않았습니다.'})
    status('정답형 두 조건 학습과 개발 평가 완료', 결과=report, 결과파일=str(JOB / 'hard_comparison.json'))
    print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 학교 GPU 비교 작업이 실행 중입니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
