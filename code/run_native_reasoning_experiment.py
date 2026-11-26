from pathlib import Path
from collections import Counter
from datetime import datetime
import ast
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback
ROOT = Path('/shared_dir/mmdl')
JOB = Path(__file__).resolve().parent
DATA = Path('/shared_dir/mmdl/data/native_reasoning_pair_20261119_232833_892795')
PREVIOUS = ROOT / 'checkpoints/expanded_lr_20261116_223345_306888'
REPO = ROOT / 'code/Qwen3-VL/qwen-vl-finetune'
REFERENCE = PREVIOUS / 'generation_eval_20261116_231128_946555'
TAGS = ['scienceqa_train', 'cosyn_circuit', 'cosyn_music', 'cosyn_chemical']
HASHES = {'answer_control': '25216d99a5fb78a9fa9f129ee8e7798dee11b67d38d2be7c43b46e2230971016', 'native_reasoning': '284ffeaf4d187141a7947bab13b196fddd15e938f67e127ca437839183669676'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def snapshot(path, source):
    compile(source, str(path), 'exec')
    assert all((ord(char) < 128 or 44032 <= ord(char) <= 55203 or ord(char) == 183 for char in source)), '코드 문자 검사에 실패했습니다.'
    if path.exists():
        assert path.read_text(encoding='utf-8') == source, f'기존 코드와 다릅니다: {path}'
    else:
        path.write_text(source, encoding='utf-8')

def set_argument(arguments, name, value):
    if name in arguments:
        arguments[arguments.index(name) + 1] = str(value)
    else:
        arguments.extend([name, str(value)])

def evaluate_code(name, adapter, output):
    source = (PREVIOUS / 'evaluate_expanded_dev.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    changed = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if 'experiments' in names:
            node.value = ast.parse(repr([(name, str(adapter))]), mode='eval').body
            pair = node.value.elts[0]
            pair.elts[1] = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[pair.elts[1]], keywords=[])
            changed.add('experiments')
        if 'output' in names:
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(output))], keywords=[])
            changed.add('output')
    assert changed == {'experiments', 'output'}, '기존 평가 코드 구조가 예상과 다릅니다.'
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'mkdir') and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'output'):
            for item in node.keywords:
                if item.arg == 'exist_ok':
                    item.value = ast.Constant(True)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'

def main():
    plan = read(DATA / 'preparation_summary.json')
    for name, expected_hash in HASHES.items():
        folder = DATA / name
        assert sha(folder / 'train.json') == expected_hash, f'{name}: 학습 데이터가 변경됐습니다.'
        assert sha(folder / 'internal_dev.json') == plan['개발 데이터 SHA256']
        assert sha(folder / 'internal_dev_provenance.json') == plan['개발 원본 기록 SHA256']
        assert len(read(folder / 'train.json')) == 4000
    original = Path(plan['기존 A 폴더'])
    assert sha(original / 'internal_dev.json') == plan['개발 데이터 SHA256']
    assert sha(original / 'internal_dev_provenance.json') == plan['개발 원본 기록 SHA256']
    save(JOB / 'experiment_plan.json', {'데이터': plan, '학습률': 0.0002, 'rank': 8, 'alpha': 16, '단계': 500, '시드': 42, '추가 teacher': False})
    (ROOT / 'checkpoints/latest_native_reasoning_comparison.txt').write_text(str(JOB), encoding='utf-8')
    snapshot(JOB / 'entry.py', (PREVIOUS / 'expanded_entry.py').read_text(encoding='utf-8'))
    (JOB / 'requirements_training.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    (JOB / 'code_commit.txt').write_text(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True), encoding='utf-8')
    environment = os.environ.copy()
    environment.update({'PILOT_REPO': str(REPO), 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'})
    predictions = {'base': read(REFERENCE / 'base_predictions.json'), 'A': read(REFERENCE / 'lr_2e-4_predictions.json')}
    assert len(predictions['base']) == 400
    assert len(predictions['A']) == 400
    assert sum((row['정답 여부'] for row in predictions['base'])) == 206
    assert sum((row['정답 여부'] for row in predictions['A'])) == 262
    for name in HASHES:
        folder = JOB / name
        folder.mkdir(exist_ok=True)
        environment.update({'PILOT_DATA': str(DATA / name), 'CHECK_DATA': '1'})
        arguments = [line for line in (PREVIOUS / 'lr_2e-4/training_arguments.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        changes = {'--output_dir': folder, '--lora_r': 8, '--lora_alpha': 16, '--learning_rate': '2e-4', '--max_steps': 500, '--seed': 42, '--data_seed': 42, '--run_name': f'format_{name}'}
        for key, value in changes.items():
            set_argument(arguments, key, value)
        argument_text = '\n'.join(arguments) + '\n'
        argument_path = folder / 'training_arguments.txt'
        if argument_path.exists():
            assert argument_path.read_text(encoding='utf-8') == argument_text
        else:
            argument_path.write_text(argument_text, encoding='utf-8')
        complete = (folder / 'pilot_result.json').is_file() and (folder / 'adapter_model.safetensors').is_file() and (read(folder / 'pilot_result.json').get('완료 단계') == 500)
        if not complete:
            status('토큰 검사와 학습 중', 구성=name, 로그=str(folder / 'training.log'))
            with (folder / 'training.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1', str(JOB / 'entry.py'), *arguments], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        config = read(folder / 'adapter_config.json')
        assert config['r'] == 8
        assert config['lora_alpha'] == 16
        assert read(folder / 'trainer_state.json')['global_step'] == 500
        assert read(folder / 'pilot_result.json')['완료 단계'] == 500
        output = JOB / f'evaluation_{name}'
        evaluator = JOB / f'evaluate_{name}.py'
        snapshot(evaluator, evaluate_code(name, folder, output))
        prediction_path = output / f'{name}_predictions.json'
        complete = prediction_path.is_file() and (output / 'comparison.json').is_file() and (len(read(prediction_path)) == 400)
        if not complete:
            status('개발 400개 생성 평가 중', 구성=name, 로그=str(folder / 'evaluation.log'))
            with (folder / 'evaluation.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', str(evaluator)], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        predictions[name] = read(prediction_path)
    report = []
    dev_originals = read(original / 'internal_dev_provenance.json')
    for name, rows in predictions.items():
        assert len(rows) == 400
        scores = {tag: Counter() for tag in TAGS}
        recovered = 0
        regressed = 0
        gained = 0
        lost = 0
        missing = 0
        exhausted = 0
        for index, row in enumerate(rows):
            baseline = predictions['base'][index]
            reference = predictions['A'][index]
            assert row['번호'] == index
            assert all((row[key] == baseline[key] == reference[key] for key in ['출처', '원본 행 번호', 'QA 번호', '기준 정답']))
            assert all((row[key] == dev_originals[index][key] for key in ['출처', '원본 행 번호', 'QA 번호']))
            correct = bool(row['정답 여부'])
            scores[row['출처']]['total'] += 1
            scores[row['출처']]['correct'] += int(correct)
            recovered += int(not baseline['정답 여부'] and correct)
            regressed += int(baseline['정답 여부'] and (not correct))
            gained += int(not reference['정답 여부'] and correct)
            lost += int(reference['정답 여부'] and (not correct))
            missing += int(not row['추출한 답'])
            exhausted += int(row['길이 한도 도달'])
        source_scores = {tag: {'문항 수': counts['total'], '정답 수': counts['correct'], '정확도': counts['correct'] / counts['total']} for tag, counts in scores.items()}
        correct_count = sum((row['정답 여부'] for row in rows))
        report.append({'구성': name, '정답 수': correct_count, '전체': 400, '정확도': correct_count / 400, '출처 평균 정확도': sum((item['정확도'] for item in source_scores.values())) / 4, '순정 오답에서 회복': recovered, '순정 정답에서 퇴보': regressed, 'A 오답에서 회복': gained, 'A 정답에서 퇴보': lost, '답 형식 누락': missing, '길이 한도 도달': exhausted, '출처별 결과': source_scores})
    save(JOB / 'format_comparison.json', {'결과': report, '주의': '같은 개발 데이터에서 선택하는 탐색 실험입니다. 독립 평가 결과가 아닙니다.'})
    status('두 지도 형식의 학습과 개발 평가 완료', 결과=report, 결과파일=str(JOB / 'format_comparison.json'))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 비교 작업이 실행 중입니다. 중복 실행을 중단합니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
