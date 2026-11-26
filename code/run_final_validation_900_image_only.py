from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import fcntl
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_independent_900 as helper
from frozen_ensemble_rule import choose_answer, GROUPS

JOB = PROJECT / 'logs/final_validation_900_image_only_20261121'
PREVIOUS = PROJECT / 'logs/final_validation_900_20261121'
IMAGE_WRAPPER = PROJECT / 'scripts/eval_mmmu_image_only.py'
RESULTS = PROJECT / 'results/final_validation_900_20261121'
FULL = PROJECT / 'logs/full_test_10500_20261121'
MODELS = {name: helper.MODELS[name] for name in ('base', 'A', 'R20', 'hard_answer', 'hard_reasoning')}
OLD = {
    'base': PROJECT / 'results_vllm_v2',
    'A': PROJECT / 'results/recipe_a_lr_2e-4_step500_retry_20261117_004730',
    'R20': PROJECT / 'results/mmmu_candidates_20261120_004215_155629/R20',
    'hard_answer': PROJECT / 'results/mmmu_hard_20261121_043545_392882/hard_answer',
    'hard_reasoning': PROJECT / 'results/mmmu_hard_20261121_043545_392882/hard_reasoning',
}
read, save, sha, fixed = helper.read, helper.save, helper.sha, helper.fixed


def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})


def live_command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def group_members(group):
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            values = path.read_text().rsplit(')', 1)[1].split()
            if int(values[2]) == group and values[0] != 'Z':
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_full_test():
    path = FULL / 'launch.json'
    if not path.exists():
        print('전체 Test 실행 기록이 없습니다.', flush=True)
        return
    launch = read(path)
    pid = int(launch['pid'])
    expected = str(FULL / 'run.py').encode()
    command = live_command(pid)
    if expected not in command:
        leftovers = group_members(pid)
        assert not leftovers, f'전체 Test 부모 프로세스는 없지만 같은 그룹이 남아 있습니다. 자동 중단하지 않습니다: {leftovers}'
        print('전체 Test 실행 프로세스가 이미 종료됐습니다.', flush=True)
        return
    assert launch['script'] == str(FULL / 'run.py')
    assert pid > 1 and pid != os.getpid()
    assert os.getpgid(pid) == pid and os.getsid(pid) == pid, '전체 Test 전용 프로세스 그룹이 아닙니다. 자동 중단하지 않습니다.'
    save(JOB / 'full_test_stop_request.json', {'pid': pid, 'script': str(FULL / 'run.py'), 'requested_at': datetime.now().isoformat(), 'scope': 'Only the verified full Test process group. Preserve all files.'})
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not group_members(pid):
            break
        time.sleep(0.5)
    if group_members(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(2)
    assert not group_members(pid), '전체 Test 프로세스 그룹의 종료를 확인하지 못했습니다.'
    state = read(FULL / 'status.json') if (FULL / 'status.json').exists() else {}
    save(FULL / 'status.json', {**state, '상태': '사용자 요청으로 중단', '갱신 시각': datetime.now().isoformat(), '사유': '같은 Validation 900문항의 최종 재현 평가로 전환. 기존 파일은 보존.'})
    print('전체 Test 작업만 중단했습니다. 기존 파일은 보존했습니다.', flush=True)


def stop_previous_validation():
    path = PREVIOUS / 'launch.json'
    if not path.exists():
        print('이전 Validation 실행 기록이 없습니다.', flush=True)
        return
    launch = read(path)
    pid = int(launch['pid'])
    expected = str(PREVIOUS / 'run.py').encode()
    command = live_command(pid)
    if expected not in command:
        leftovers = group_members(pid)
        assert not leftovers, f'이전 Validation 부모 프로세스는 없지만 같은 그룹이 남아 있습니다. 자동 중단하지 않습니다: {leftovers}'
        print('이전 Validation 실행 프로세스가 이미 종료됐습니다.', flush=True)
        return
    assert launch['script'] == str(PREVIOUS / 'run.py')
    assert pid > 1 and pid != os.getpid()
    assert os.getpgid(pid) == pid and os.getsid(pid) == pid, '이전 Validation 전용 프로세스 그룹이 아닙니다. 자동 중단하지 않습니다.'
    save(JOB / 'previous_validation_stop_request.json', {'pid': pid, 'script': str(PREVIOUS / 'run.py'), 'requested_at': datetime.now().isoformat(), 'scope': 'Only the verified full Test process group. Preserve all files.'})
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not group_members(pid):
            break
        time.sleep(0.5)
    if group_members(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(2)
    assert not group_members(pid), '이전 Validation 프로세스 그룹의 종료를 확인하지 못했습니다.'
    state = read(PREVIOUS / 'status.json') if (PREVIOUS / 'status.json').exists() else {}
    save(PREVIOUS / 'status.json', {**state, '상태': '사용자 요청으로 중단', '갱신 시각': datetime.now().isoformat(), '사유': '같은 Validation 900문항의 최종 재현 평가로 전환. 기존 파일은 보존.'})
    print('이전 Validation 작업만 중단했습니다. 기존 파일은 보존했습니다.', flush=True)


def gpu_available():
    for _ in range(24):
        used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if used < 2000:
            return
        status('GPU 메모리 반환 대기 중', GPU메모리_MiB=used)
        time.sleep(5)
    raise RuntimeError('다른 GPU 작업이 실행 중이거나 메모리가 반환되지 않았습니다. 다른 프로세스를 강제로 종료하지 않습니다.')


def load_results(folder, subjects, reference=None):
    values = {}
    for subject in subjects:
        rows = read(folder / (subject + '.json'))
        assert len(rows) == 30 and len({row['id'] for row in rows}) == 30
        for row in rows:
            assert row['subject'] == subject and row['id'] not in values
            if reference is not None:
                old = reference[row['id']]
                assert all(row[key] == old[key] for key in ('subject', 'answer', 'is_mcq'))
            values[row['id']] = row
    assert len(values) == 900
    if reference is not None:
        assert set(values) == set(reference)
    return values


def compare(subjects, reference):
    data = {name: load_results(RESULTS / name, subjects, reference) for name in MODELS}
    ids = sorted(reference)
    for group, names in GROUPS.items():
        votes = []
        for candidate in ids:
            predicted, selected = choose_answer({name: data[name][candidate]['predicted'] for name in names}, group)
            votes.append({'id': candidate, 'predicted': predicted, 'selected_model': selected})
        save(RESULTS / 'votes' / (group + '.json'), votes)
        data[group] = {row['id']: {**data[row['selected_model']][row['id']], 'selected_model': row['selected_model']} for row in votes}
    report = {}
    base = [data['base'][candidate]['correct'] for candidate in ids]
    for name, rows in data.items():
        correct = [rows[candidate]['correct'] for candidate in ids]
        counts = Counter(row['subject'] for row in rows.values() if row['correct'])
        report[name] = {'correct': sum(correct), 'total': 900, 'accuracy': sum(correct) / 900,
                        'baseline_difference_percentage_points': (sum(correct) - sum(base)) / 9,
                        'baseline_comparison': helper.pair(base, correct),
                        'subjects': {subject: {'correct': counts[subject], 'total': 30} for subject in subjects}}
        save(RESULTS / 'scored' / (name + '.json'), [rows[candidate] for candidate in ids])
        if name in OLD:
            previous = load_results(OLD[name], subjects, reference)
            report[name]['reproduction'] = {
                'previous_correct': sum(row['correct'] for row in previous.values()),
                'changed_correctness': sum(rows[candidate]['correct'] != previous[candidate]['correct'] for candidate in ids),
                'identical_predictions': sum(rows[candidate]['predicted'] == previous[candidate]['predicted'] for candidate in ids),
                'identical_raw_outputs': sum(rows[candidate]['raw_output'] == previous[candidate]['raw_output'] for candidate in ids)}
    save(RESULTS / 'comparison.json', {'results': report, 'protocol': read(JOB / 'protocol.json'),
         'interpretation': 'Final reproduction on the same previously used Validation 900. This is not an independent evaluation. Frozen groups and tie rules; votes use predictions only. Custom MMMU grading is retained. Five-model inference cost differs from a single model. New scores are measured, not forced to equal previous results.'})
    status('Validation 900개 최종 재현 평가 완료', 결과={name: value['correct'] for name, value in report.items()}, 결과파일=str(RESULTS / 'comparison.json'))


def run():
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('최종 Validation 평가가 이미 실행 중입니다.')
        try:
            status('이전 평가 초기화 작업 중단 확인 중')
            stop_previous_validation()
            stop_full_test()
            helper.configure()
            for package, version in (('torch', '2.8.0'), ('transformers', '4.57.6'), ('vllm', '0.11.0')):
                assert importlib.metadata.version(package).split('+')[0] == version
            assert sha(helper.EVALUATOR) == helper.EVALUATOR_SHA
            protocol = read(helper.DATA / 'candidate_protocol.json')
            assert sha(PROJECT / 'scripts/frozen_ensemble_rule.py') == protocol['voting_code_sha256']
            subjects = sorted(read(OLD['base'] / '_summary.json')['subject_accuracy'])
            assert len(subjects) == 30
            reference = load_results(OLD['base'], subjects)
            binary = read(PROJECT / 'logs/independent_900_20261121/model_binary_lock.json')
            fixed(JOB / 'protocol.json', {
                'models': {name: str(path) for name, path in MODELS.items()},
                'groups': {name: list(names) for name, names in GROUPS.items()},
                'voting_code_sha256': protocol['voting_code_sha256'], 'inference': protocol['inference'],
                'evaluator_sha256': helper.EVALUATOR_SHA, 'runner_sha256': sha(Path(__file__)),
                'image_only_wrapper_sha256': sha(IMAGE_WRAPPER),
                'resource_change': {'limit_mm_per_prompt': {'image': 8, 'video': 0}},
                'comparison_note': 'Disable unused video profiling for every candidate; original evaluator bytes and generation settings remain fixed. Measure baseline reproduction instead of assuming exact output identity.',
                'original_validation_files_sha256': {subject: sha(OLD['base'] / (subject + '.json')) for subject in subjects},
                'dataset_revision': '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',
                'dataset_split': 'validation', 'samples': 900, 'reuse_previous_predictions': False,
                'reporting': 'Same Validation reproduction; independent Test 900 is a separate supplementary result.'})
            for name, folder in MODELS.items():
                status('고정 모델 파일 검사 중', 모델=name)
                assert str(folder) == binary[name]['path']
                assert sha(folder / 'config.json') == binary[name]['config_sha256']
                for filename, expected in binary[name]['weights'].items():
                    assert sha(folder / filename) == expected
                target = RESULTS / name
                target.mkdir(parents=True, exist_ok=True)
                for subject in subjects:
                    path = target / (subject + '.json')
                    if path.exists():
                        rows = read(path)
                        assert len(rows) == 30 and {row['id'] for row in rows} == {key for key, row in reference.items() if row['subject'] == subject}
                        assert all(row['answer'] == reference[row['id']]['answer'] for row in rows)
                command = [sys.executable, '-u', str(IMAGE_WRAPPER), '--model_path', str(folder),
                           '--model_revision', 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
                           '--data_root', 'MMMU/MMMU', '--data_revision', '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68', '--out_dir', str(target)]
                fixed(JOB / (name + '_command.json'), command)
                if not (target / '_summary.json').exists():
                    gpu_available()
                    logpath = JOB / (name + '.log')
                    status('Validation 900개 최종 재현 평가 중', 모델=name, 로그=str(logpath))
                    with logpath.open('a', encoding='utf-8') as log:
                        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert sha(helper.EVALUATOR) == helper.EVALUATOR_SHA
                summary = read(target / '_summary.json')
                for key, expected in {'model_path': str(folder), 'max_new_tokens': 512, 'min_pixels': 200704, 'max_pixels': 802816, 'seed': 3407, 'presence_penalty': 1.5, 'backend': 'vllm'}.items():
                    assert summary[key] == expected, f'평가 설정 불일치: {key}'
                load_results(target, subjects, reference)
            compare(subjects, reference)
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise


def launch():
    JOB.mkdir(parents=True, exist_ok=True)
    launch_path = JOB / 'launch.json'
    if launch_path.exists():
        info = read(launch_path)
        if str(JOB / 'run.py').encode() in live_command(info['pid']):
            print('최종 Validation 평가가 이미 실행 중입니다. 확인 명령만 실행하세요.')
            return
    target = JOB / 'run.py'
    source = Path(__file__).read_bytes()
    if target.exists():
        assert target.read_bytes() == source, '기존 최종 평가 실행 파일과 다릅니다.'
    else:
        target.write_bytes(source)
    with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(target), '--run'], cwd=PROJECT,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(launch_path, {'pid': process.pid, 'script': str(target)})
    print('Validation 900개 최종 평가 실행 요청 완료')
    print('실행 과정에서 이전 Validation 초기화 작업과 남은 전체 Test 작업만 확인해 중단합니다.')
    print('프로세스 번호:', process.pid)
    print('로그:', JOB / 'runner.log')


def monitor(watch):
    while True:
        state = read(JOB / 'status.json') if (JOB / 'status.json').exists() else {'상태': '아직 상태 파일 없음'}
        print('\n확인 시각:', datetime.now().isoformat(timespec='seconds'))
        print(json.dumps(state, ensure_ascii=False, indent=2))
        for name in MODELS:
            folder = RESULTS / name
            count = sum(len(read(path)) for path in folder.glob('*.json') if not path.name.startswith('_')) if folder.exists() else 0
            print(name, ':', count, '/900개 저장')
        if (JOB / 'launch.json').exists():
            pid = read(JOB / 'launch.json')['pid']
            print('순차 실행 프로세스:', pid, '실행 중' if str(JOB / 'run.py').encode() in live_command(pid) else '실행되지 않음')
        path = Path(state.get('로그', str(JOB / 'runner.log')))
        if path.exists():
            print('최근 로그:', path)
            print('\n'.join(path.read_text(encoding='utf-8', errors='replace').replace('\r', '\n').splitlines()[-8:]))
        if not watch or state['상태'] in ('Validation 900개 최종 재현 평가 완료', '오류로 중단'):
            break
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.run:
        run()
    else:
        try:
            monitor(args.watch)
        except KeyboardInterrupt:
            print('\n확인 화면만 종료했습니다. 백그라운드 평가는 계속됩니다.')
