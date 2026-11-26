from pathlib import Path
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
PREVIOUS = ROOT / 'checkpoints/expanded_lr_20261116_223345_306888'
DATA = ROOT / 'data/recipe_a_answer_only_20261115_004649_140669'
REPO = ROOT / 'code/Qwen3-VL/qwen-vl-finetune'
REFERENCE = PREVIOUS / 'generation_eval_20261116_231128_946555'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(name, value):
    path = JOB / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save('status.json', {'상태': state, **extra})

def set_argument(arguments, name, value):
    if name in arguments:
        arguments[arguments.index(name) + 1] = str(value)
    else:
        arguments.extend([name, str(value)])

def main():
    previous_hashes = read(PREVIOUS / 'data_hashes.json')
    for name in ['train.json', 'internal_dev.json']:
        current = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert current == previous_hashes[name], f'기존 데이터가 변경됐습니다: {name}'
    save('data_hashes.json', previous_hashes)
    save('experiment_plan.json', {'teacher': False, 'new_data': False, 'ranks': [4, 16], 'alphas': [8, 32], 'baseline_rank': 8, 'learning_rate': 0.0002, 'steps': 500, 'seed': 42, 'purpose': 'Compare training rank while keeping alpha/r equal to 2.'})
    entry = (PREVIOUS / 'expanded_entry.py').read_text(encoding='utf-8')
    ast.parse(entry)
    (JOB / 'entry.py').write_text(entry, encoding='utf-8')
    requirements = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (JOB / 'requirements_training.txt').write_text(requirements, encoding='utf-8')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True)
    (JOB / 'code_commit.txt').write_text(commit, encoding='utf-8')
    environment = os.environ.copy()
    environment.update({'PILOT_REPO': str(REPO), 'PILOT_DATA': str(DATA), 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1', 'CHECK_DATA': '0'})
    for rank in [4, 16]:
        folder = JOB / f'rank{rank}'
        folder.mkdir(exist_ok=True)
        arguments = (PREVIOUS / 'lr_2e-4/training_arguments.txt').read_text(encoding='utf-8').splitlines()
        changes = {'--output_dir': folder, '--lora_r': rank, '--lora_alpha': rank * 2, '--learning_rate': '2e-4', '--max_steps': 500, '--seed': 42, '--data_seed': 42}
        for name, value in changes.items():
            set_argument(arguments, name, value)
        (folder / 'training_arguments.txt').write_text('\n'.join(arguments) + '\n', encoding='utf-8')
        status('학습 중', rank=rank, 로그=str(folder / 'training.log'))
        with (folder / 'training.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1', str(JOB / 'entry.py'), *arguments], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        config = read(folder / 'adapter_config.json')
        assert config['r'] == rank
        assert config['lora_alpha'] == rank * 2
        assert read(folder / 'pilot_result.json')['완료 단계'] == 500
        assert (folder / 'adapter_model.safetensors').is_file()
    source = (PREVIOUS / 'evaluate_expanded_dev.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    changed = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if 'experiments' in names:
            node.value = ast.parse(repr([('rank4', str(JOB / 'rank4')), ('rank16', str(JOB / 'rank16'))]), mode='eval').body
            for item in node.value.elts:
                item.elts[1] = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[item.elts[1]], keywords=[])
            changed.add('experiments')
        if 'output' in names:
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(value=str(JOB / 'evaluation'))], keywords=[])
            changed.add('output')
    assert changed == {'experiments', 'output'}
    ast.fix_missing_locations(tree)
    evaluation = ast.unparse(tree) + '\n'
    ast.parse(evaluation)
    assert all((ord(char) < 128 or 44032 <= ord(char) <= 55203 or ord(char) == 183 for char in evaluation)), '평가 코드 문자 검사에 실패했습니다.'
    (JOB / 'evaluate.py').write_text(evaluation, encoding='utf-8')
    status('개발 400개 생성 평가 중', 로그=str(JOB / 'evaluation.log'))
    with (JOB / 'evaluation.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-u', str(JOB / 'evaluate.py')], cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    base = read(REFERENCE / 'base_predictions.json')
    baseline = read(REFERENCE / 'lr_2e-4_predictions.json')
    assert len(base) == len(baseline) == 400
    assert sum((row['정답 여부'] for row in base)) == 206
    assert sum((row['정답 여부'] for row in baseline)) == 262
    comparisons = [('rank8', baseline), ('rank4', read(JOB / 'evaluation/rank4_predictions.json')), ('rank16', read(JOB / 'evaluation/rank16_predictions.json'))]
    report = []
    for label, values in comparisons:
        assert len(values) == 400
        recovered = regressed = 0
        for index, value in enumerate(values):
            assert value['번호'] == base[index]['번호'] == index
            assert all((value[key] == base[index][key] for key in ['기준 정답', '출처', '원본 행 번호', 'QA 번호']))
            recovered += int(not base[index]['정답 여부'] and value['정답 여부'])
            regressed += int(base[index]['정답 여부'] and (not value['정답 여부']))
        report.append({'설정': label, '정답 수': sum((row['정답 여부'] for row in values)), '전체': 400, '순정 오답에서 회복': recovered, '순정 정답에서 퇴보': regressed})
    save('rank_comparison.json', {'결과': report})
    status('두 학습과 개발 평가 완료', 결과=report, 결과파일=str(JOB / 'rank_comparison.json'))
if __name__ == '__main__':
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (ROOT / 'checkpoints/latest_rank_comparison.txt').write_text(str(JOB), encoding='utf-8')
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
