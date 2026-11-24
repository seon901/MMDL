from pathlib import Path
from datetime import datetime
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
JOB = PROJECT / 'logs/A_reproduction_20261120'
RESULT = PROJECT / 'results/A_reproduction_20261120'
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_mmmu_candidates as batch
from resume_mmmu_candidates import configure

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def main():
    with (batch.JOB / 'batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 MMMU 배치가 실행 중입니다. 중복 실행하지 않습니다.')
        try:
            status('기준 A 재현 평가 준비 중')
            configure()
            assert batch.digest(batch.EVALUATOR) == batch.EVALUATOR_SHA
            for package, version in [('torch', '2.8.0'), ('transformers', '4.57.6'), ('vllm', '0.11.0')]:
                assert importlib.metadata.version(package).split('+')[0] == version
            subjects = list(read(batch.A_RESULT / '_summary.json')['subject_accuracy'])
            old = batch.results(batch.A_RESULT, subjects)
            assert sum(row['correct'] for row in old.values()) == 501
            model = batch.SETTINGS
            assert model.is_dir()
            weights = sorted(set(read(model / 'model.safetensors.index.json')['weight_map'].values()))
            metadata = {'기준 모델': str(model), '기준 모델 가중치 SHA256': {name: batch.digest(model / name) for name in weights}, '평가 코드 SHA256': batch.EVALUATOR_SHA, 'CUDA_HOME': os.environ['CUDA_HOME'], 'VLLM_USE_FLASHINFER_SAMPLER': os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'), '과거 A 정답 수': 501}
            metadata_path = JOB / 'reproduction_conditions.json'
            if metadata_path.exists():
                assert read(metadata_path) == metadata
            else:
                save(metadata_path, metadata)
            RESULT.mkdir(parents=True, exist_ok=True)
            for subject in subjects:
                path = RESULT / f'{subject}.json'
                if path.exists():
                    values = read(path)
                    expected = {key for key, row in old.items() if row['subject'] == subject}
                    assert len(values) == 30 and {row['id'] for row in values} == expected
            command = [sys.executable, '-u', str(batch.EVALUATOR), '--model_path', str(model), '--model_revision', batch.BASE_REVISION, '--data_root', 'MMMU/MMMU', '--data_revision', batch.DATA_REVISION, '--out_dir', str(RESULT)]
            save(JOB / 'evaluation_command.json', command)
            if not (RESULT / '_summary.json').exists():
                status('기준 A의 MMMU 재현 평가 중', 로그=str(JOB / 'evaluation.log'), 결과폴더=str(RESULT))
                with (JOB / 'evaluation.log').open('a', encoding='utf-8') as log:
                    subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert batch.digest(batch.EVALUATOR) == batch.EVALUATOR_SHA
            summary = read(RESULT / '_summary.json')
            for key, expected in {'model_path':str(model), 'max_new_tokens':512, 'min_pixels':200704, 'max_pixels':802816, 'seed':3407, 'presence_penalty':1.5, 'backend':'vllm'}.items():
                assert summary[key] == expected
            current = batch.results(RESULT, subjects)
            assert set(current) == set(old)
            r20 = batch.results(batch.RESULT_ROOT / 'R20', subjects)
            report = {
                '과거 A 정답 수': 501,
                '현재 A 정답 수': sum(row['correct'] for row in current.values()),
                '과거 A와 정오답이 달라진 문항 수': sum(old[key]['correct'] != row['correct'] for key, row in current.items()),
                '과거 A와 추출 답이 같은 문항 수': sum(old[key]['predicted'] == row['predicted'] for key, row in current.items()),
                '과거 A와 전체 출력이 같은 문항 수': sum(old[key]['raw_output'] == row['raw_output'] for key, row in current.items()),
                '현재 A의 과거 A 대비 변화': batch.paired(old, current),
                'R20 정답 수': sum(row['correct'] for row in r20.values()),
                'R20의 현재 A 대비 변화': batch.paired(current, r20),
                '평가 코드 SHA256': batch.EVALUATOR_SHA,
            }
            save(JOB / 'reproduction_comparison.json', report)
            status('기준 A 재현 평가 완료', 결과=report, 결과파일=str(JOB / 'reproduction_comparison.json'))
            print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc(), 로그=str(JOB / 'evaluation.log'))
            raise

def launch():
    assert PROJECT.is_dir(), '개인 컴퓨터 WSL에서 실행해 주세요.'
    JOB.mkdir(parents=True, exist_ok=True)
    target = JOB / 'run.py'
    source = Path(__file__).read_bytes()
    if target.exists():
        assert target.read_bytes() == source, '기존 실행 파일과 다릅니다.'
    else:
        target.write_bytes(source)
    with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
        log.write('\n=== 기존 A 재현 평가 실행 요청 ===\n')
        log.flush()
        process = subprocess.Popen([sys.executable, '-u', str(target)], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('=== 기존 A 재현 평가 실행 요청 완료 ===')
    print('프로세스 번호:', process.pid)
    print('상태 파일:', JOB / 'status.json')
    print('평가 로그:', JOB / 'evaluation.log')

if __name__ == '__main__':
    os.environ['PATH'] = str(PROJECT / 'venv_vllm/bin') + os.pathsep + os.environ.get('PATH', '')
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    arguments = parser.parse_args()
    if arguments.launch:
        launch()
    else:
        main()