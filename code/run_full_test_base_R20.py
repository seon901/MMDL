from pathlib import Path
from datetime import datetime
import argparse
import ast
import fcntl
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
SCRIPTS = PROJECT / 'scripts'
SOURCE = SCRIPTS / 'run_full_test_base_R20.py'
sys.path.insert(0, str(SCRIPTS))
import run_full_test_10500 as g

JOB = PROJECT / 'logs/full_test_base_R20_20261121'
RESULTS = PROJECT / 'results/full_test_base_R20_20261121'
INPUT_SHA = '4e1987d8faa991def1c3c0e9fe05f4c5cc26d60a517b1aa6b37105dbfbd95d3d'
ANSWER_SHA = '45b639a937017d3d71f81f610df3a1471debe1887c5c1d24644fec1b33059d5c'
g.JOB = JOB
g.RESULTS = RESULTS
g.MODELS = {name: g.helper.MODELS[name] for name in ('base', 'R20')}
g.GROUPS = {}
g.__file__ = str(Path(__file__).resolve())


def code_lock():
    names = ['run_full_test_base_R20.py', 'run_full_test_10500.py',
             'run_independent_900.py', 'resume_mmmu_candidates.py', 'eval_mmmu_vllm.py']
    values = {name: g.sha(SCRIPTS / name) for name in names}
    for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'chat_template.json', 'merges.txt', 'vocab.json', 'video_preprocessor_config.json']:
        path = g.helper.SETTINGS / name
        if path.exists():
            values['processor/' + name] = g.sha(path)
    return values


g.code_lock = code_lock


def protocol():
    return {
        'name': 'MMMU full Test 10500, base versus R20, fixed local grading',
        'models': {name: str(folder) for name, folder in g.MODELS.items()},
        'repository': 'MMMU/MMMU',
        'revision': '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',
        'split': 'test', 'total': 10500, 'batch_size': g.BATCH,
        'input_sha256': INPUT_SHA,
        'inference': g.read(g.REFERENCE / 'candidate_protocol.json')['inference'],
        'scope': 'Only base and R20. No ensemble. All official Test IDs. No previous model outputs are reused.',
        'context': 'Before inference, inspect all original and MCQ retry inputs. Same fixed context for both models, between 4096 and 8192 tokens, reserving 512 output tokens.',
        'grading': 'Reuse the frozen custom MMMU parser, retry and open-answer grader. This is not an official leaderboard score.',
        'limitations': 'Test includes previously evaluated IDs. Report full Test and prior-ID subsets separately. No claim of exhaustive training-overlap exclusion.',
        'official_answer_sha256': ANSWER_SHA,
    }


def verify_models():
    previous = g.read(PROJECT / 'logs/independent_900_20261121/model_binary_lock.json')
    selected = {}
    for name, folder in g.MODELS.items():
        expected = previous[name]
        assert folder == Path(expected['path']), f'모델 경로가 다릅니다: {name}'
        assert g.sha(folder / 'config.json') == expected['config_sha256']
        for filename, digest in expected['weights'].items():
            g.status('고정 모델 파일 검사 중', 모델=name, 파일=filename)
            assert g.sha(folder / filename) == digest, f'모델 가중치가 다릅니다: {name}/{filename}'
        selected[name] = expected
    g.fixed(JOB / 'model_binary_lock.json', selected)


def preflight(rows):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(g.helper.SETTINGS, local_files_only=True)
    functions = g.helper.primary_functions()
    binding = {'code': code_lock(), 'input_sha256': INPUT_SHA}
    cache_path = JOB / 'token_lengths.json'
    if cache_path.exists():
        cache = g.read(cache_path)
        assert cache['binding'] == binding, '입력 길이 검사 코드나 데이터가 바뀌었습니다.'
        records = cache['rows']
    else:
        records = []
    assert len(records) <= len(rows)
    assert all(item['id'] == row['id'] for item, row in zip(records, rows))
    maximum = max((r['max_prompt_tokens'] for r in records), default=0)
    started, initial = time.monotonic(), len(records)
    g.status('전체 Test 입력 길이 검사 중', 완료=initial, 전체=len(rows), 최대입력토큰=maximum)
    for index in range(initial, len(rows)):
        row = rows[index]
        row_max = 0
        for retry in ([False, True] if row['question_type'] == 'multiple-choice' else [False]):
            value, _, _, owners = g.helper.request(row, processor, functions, retry)
            try:
                kwargs = {'text': [value['prompt']], 'return_tensors': 'pt'}
                images = value.get('multi_modal_data', {}).get('image')
                if images:
                    kwargs['images'] = images
                length = int(processor(**kwargs)['input_ids'].shape[-1])
                row_max = max(row_max, length)
            finally:
                g.helper.close_images(owners)
        maximum = max(maximum, row_max)
        records.append({'id': row['id'], 'max_prompt_tokens': row_max})
        if len(records) % 100 == 0 or len(records) == len(rows):
            g.save(cache_path, {'binding': binding, 'rows': records})
            done = len(records) - initial
            remaining = (time.monotonic() - started) / max(1, done) * (len(rows) - len(records))
            g.status('전체 Test 입력 길이 검사 중', 완료=len(records), 전체=len(rows),
                     최대입력토큰=maximum, 입력검사_예상남은분=round(remaining / 60, 1))
            print('입력 길이 검사:', len(records), '/', len(rows),
                  '검사 예상 남은 시간:', round(remaining / 60, 1), '분', flush=True)
    context = max(4096, math.ceil((maximum + 512) / 1024) * 1024)
    assert context <= 8192, '전체 Test의 입력이 사전 문맥 기준 8192를 초과했습니다. 추론은 시작하지 않았습니다.'
    g.fixed(JOB / 'execution_lock.json', {
        'code': code_lock(), 'input_sha256': INPUT_SHA,
        'protocol_sha256': g.sha(JOB / 'protocol.json'),
        'max_model_len': context, 'max_prompt_tokens': maximum,
        'grading_file_sha256': ANSWER_SHA,
        'all_models_same_context': True, 'labels_loaded_only_after_prediction_seal': True,
    })


def wait_gpu():
    while True:
        output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], text=True)
        used = int(output.splitlines()[0])
        if used < 2000:
            return
        g.status('기존 GPU 작업 종료 대기 중', GPU메모리_MiB=used)
        print('기존 GPU 작업을 종료하지 않고 기다립니다. GPU 메모리:', used, 'MiB', flush=True)
        time.sleep(30)


def run():
    g.helper.configure()
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('순정과 R20의 전체 Test 평가가 이미 실행 중입니다.')
        try:
            assert Path(__file__).read_bytes() == SOURCE.read_bytes(), '실행 사본과 원본 코드가 다릅니다.'
            for name, version in [('torch', '2.8.0'), ('transformers', '4.57.6'), ('vllm', '0.11.0')]:
                assert importlib.metadata.version(name).split('+')[0] == version, f'환경 확인 필요: {name}'
            assert g.sha(g.helper.EVALUATOR) == g.helper.EVALUATOR_SHA
            assert g.sha(g.DATA / 'blind_inputs.json') == INPUT_SHA
            assert g.sha(g.REFERENCE / 'official_answer_dict_test.json') == ANSWER_SHA
            rows = g.inputs()
            assert len(rows) == 10500
            g.fixed(JOB / 'protocol.json', protocol())
            verify_models()
            if not (JOB / 'execution_lock.json').exists():
                preflight(rows)
            execution = g.read(JOB / 'execution_lock.json')
            assert execution['code'] == code_lock()
            assert execution['protocol_sha256'] == g.sha(JOB / 'protocol.json')
            assert execution['input_sha256'] == INPUT_SHA
            for name in g.MODELS:
                if all((RESULTS / 'blind_predictions' / name / (key + '.json')).exists()
                       for key, _ in g.chunks(rows)):
                    g.collect(name, rows)
                    continue
                wait_gpu()
                log = JOB / (name + '.log')
                g.status('전체 Test 모델 로딩 중', 모델=name, 로그=str(log))
                with log.open('a', encoding='utf-8') as stream:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name],
                                   cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            g.status('두 모델 예측 완료. 고정 채점 중')
            g.score(rows)
        except Exception:
            g.status('오류로 중단', 오류=traceback.format_exc())
            raise


def launch():
    g.helper.configure()
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = JOB / 'launch.json'
        if info.exists():
            pid = g.read(info)['pid']
            command = Path('/proc') / str(pid) / 'cmdline'
            if command.exists() and str(JOB / 'run.py').encode() in command.read_bytes().split(b'\0'):
                raise SystemExit('이미 실행 중입니다. --watch로 진행 상태를 확인하세요.')
        with (JOB / 'evaluation.lock').open('a') as evaluation_lock:
            try:
                fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('이미 평가가 실행 중입니다.')
        path = JOB / 'run.py'
        source = SOURCE.read_bytes()
        if path.exists():
            assert path.read_bytes() == source, '기존 실행 코드와 달라 자동 재개하지 않습니다.'
        else:
            path.write_bytes(source)
        with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', str(path), '--run'], cwd=PROJECT,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        g.save(info, {'pid': process.pid, 'script': str(path), 'started': datetime.now().isoformat()})
        print('순정과 R20의 Test 10500문항 평가 실행 요청 완료')
        print('모델 수: 2개. 앙상블은 실행하지 않습니다.')
        print('프로세스 번호:', process.pid)
        print('진행 확인: 같은 스크립트에 --watch를 사용하세요.')
        print('결과 폴더:', RESULTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', choices=['base', 'R20'])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.run:
        run()
    elif args.worker:
        assert code_lock() == g.read(JOB / 'execution_lock.json')['code']
        g.worker(args.worker)
    else:
        try:
            g.monitor(args.watch)
        except KeyboardInterrupt:
            print('\n확인 화면만 종료했습니다. 백그라운드 평가는 계속됩니다.')
