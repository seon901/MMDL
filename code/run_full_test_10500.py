from pathlib import Path
from datetime import datetime
from collections import Counter
from io import BytesIO
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_independent_900 as helper
from frozen_ensemble_rule import choose_answer, GROUPS

JOB = PROJECT / 'logs/full_test_10500_20261121'
RESULTS = PROJECT / 'results/full_test_10500_20261121'
DATA = PROJECT / 'evaluation_reference/full_test_10500_20261121'
REFERENCE = PROJECT / 'evaluation_reference/independent_900_20261121'
RAW = PROJECT / 'evaluation_reference/MMMU_Test/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
MODELS = {name: helper.MODELS[name] for name in ('R20', 'A', 'base', 'hard_reasoning', 'hard_answer')}
TOTAL = 10500
BATCH = 10
read, save, sha, fixed = helper.read, helper.save, helper.sha, helper.fixed
TERMINAL = ('전체 Test 평가 완료', '오류로 중단')


def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})


def code_lock():
    return {name: sha(PROJECT / 'scripts' / name) for name in
            ('run_full_test_10500.py', 'run_independent_900.py', 'frozen_ensemble_rule.py', 'eval_mmmu_vllm.py')}


def inputs():
    lock = read(DATA / 'input_lock.json')
    assert sha(DATA / 'blind_inputs.json') == lock['input_sha256']
    rows = read(DATA / 'blind_inputs.json')
    assert len(rows) == TOTAL and len({row['id'] for row in rows}) == TOTAL
    assert len(Counter(row['subject'] for row in rows)) == 30
    assert dict(Counter(row['subject'] for row in rows)) == lock['subject_counts']
    assert all(not {'answer', 'gold', 'correct', 'ground_truth', 'explanation'} & set(row) for row in rows)
    return rows


def prepare():
    import pyarrow.parquet as pq
    from PIL import Image
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = read(REFERENCE / 'candidate_protocol.json')
    assert sha(helper.EVALUATOR) == helper.EVALUATOR_SHA
    assert sha(PROJECT / 'scripts/frozen_ensemble_rule.py') == protocol['voting_code_sha256']
    fixed(DATA / 'protocol.json', {
        'models': {name: str(path) for name, path in MODELS.items()},
        'source_protocol_sha256': sha(REFERENCE / 'candidate_protocol.json'),
        'groups': {key: list(value) for key, value in GROUPS.items()}, 'inference': protocol['inference'],
        'batch_size': BATCH, 'reuse_old_predictions': False,
        'reporting': 'Full Test 10500 includes previously observed IDs. Report the full set and previously unobserved IDs separately. The latter is an ID-based supplementary subset, not a complete contamination audit.',
        'selection': 'Every official Test ID, sorted by subject and ID. No selection by model scores or gold answers.',
        'grading': 'Keep the existing custom MMMU grading implementation; do not claim an official leaderboard score.'})
    source = read(REFERENCE / 'preparation_summary.json')['원본 파일']
    assert len(source) >= 30
    columns = ['id', 'question', 'options', 'question_type'] + [f'image_{i}' for i in range(1, 8)]
    by_subject = {}
    for filename in sorted(source):
        by_subject.setdefault(filename.split('/')[0], []).append(filename)
    assert len(by_subject) == 30
    manifest = []
    for subject, filenames in sorted(by_subject.items()):
        part = DATA / 'subjects' / (subject + '.json')
        if part.exists():
            record = read(part)
            assert record['sources'] == {filename: source[filename]['sha256'] for filename in filenames}
            for filename in filenames:
                assert sha(RAW / filename) == source[filename]['sha256']
            manifest.extend(record['rows'])
            continue
        status('전체 Test 입력 준비 중', 과목=subject, 완료=len(manifest), 전체=TOTAL)
        rows = []
        for filename in filenames:
            path = RAW / filename
            assert sha(path) == source[filename]['sha256'], f'원본 파일 검증 실패: {path}'
            table = pq.read_table(path, columns=columns)
            for batch in table.to_batches(max_chunksize=16):
                for original in batch.to_pylist():
                    row = {key: original[key] for key in ('id', 'question', 'options', 'question_type')}
                    row.update(subject=subject, images={})
                    for n in range(1, 8):
                        cell = original.get(f'image_{n}')
                        if not cell:
                            continue
                        destination = DATA / 'images' / (original['id'] + f'_image_{n}.png')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if cell.get('bytes'):
                            handle = BytesIO(cell['bytes'])
                        else:
                            handle = Path(cell['path'])
                            assert handle.is_absolute() and handle.is_file(), f'이미지 경로 확인 필요: {handle}'
                        with Image.open(handle) as image:
                            image.convert('RGB').save(destination, format='PNG')
                        row['images'][f'image_{n}'] = destination.relative_to(PROJECT).as_posix()
                    row['input_text_sha256'] = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    row['image_sha256'] = {key: sha(PROJECT / value) for key, value in row['images'].items()}
                    rows.append(row)
            del table
        rows.sort(key=lambda row: row['id'])
        assert len(rows) == sum(pq.ParquetFile(RAW / filename).metadata.num_rows for filename in filenames)
        save(part, {'sources': {filename: source[filename]['sha256'] for filename in filenames}, 'rows': rows})
        manifest.extend(rows)
        print(subject, ': 입력', len(rows), '개 준비 완료', flush=True)
    assert len(manifest) == TOTAL and len({row['id'] for row in manifest}) == TOTAL
    fixed(DATA / 'blind_inputs.json', manifest)
    fixed(DATA / 'input_lock.json', {'input_sha256': sha(DATA / 'blind_inputs.json'), 'subject_counts': dict(Counter(row['subject'] for row in manifest)), 'source_records': source})
    seen = {row['id'] for row in read(REFERENCE / 'blind_inputs.json')}
    pro = PROJECT / 'evaluation_reference/MMMU_Pro/0d7426df4ccb3d8704a992fe6850abee7c262127/standard (10 options)'
    for path in sorted(pro.glob('*.parquet')):
        seen.update(pq.read_table(path, columns=['id']).column('id').to_pylist())
    assert len(list(pro.glob('*.parquet'))) == 2
    fixed(DATA / 'previously_seen_test_ids.json', sorted(seen & {row['id'] for row in manifest}))
    return manifest


def preflight(rows):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(helper.SETTINGS, local_files_only=True)
    functions = helper.primary_functions()
    maximum = 0
    for index, row in enumerate(rows):
        for retry in ([False, True] if row['question_type'] == 'multiple-choice' else [False]):
            value, _, _, owners = helper.request(row, processor, functions, retry)
            try:
                kwargs = {'text': [value['prompt']], 'return_tensors': 'pt'}
                images = value.get('multi_modal_data', {}).get('image')
                if images:
                    kwargs['images'] = images
                length = int(processor(**kwargs)['input_ids'].shape[-1])
                maximum = max(maximum, length)
            finally:
                helper.close_images(owners)
        if (index + 1) % 100 == 0:
            status('전체 Test 입력 길이 검사 중', 완료=index + 1, 전체=TOTAL, 최대입력토큰=maximum)
            print('입력 길이 검사:', index + 1, '/', TOTAL, flush=True)
    context = max(4096, math.ceil((maximum + 512) / 1024) * 1024)
    assert context <= 8192, '입력 길이가 사전 기준을 초과했습니다. 추론을 시작하지 않습니다.'
    fixed(JOB / 'execution_lock.json', {
        'code': code_lock(), 'input_sha256': sha(DATA / 'blind_inputs.json'),
        'protocol_sha256': sha(DATA / 'protocol.json'), 'max_model_len': context,
        'max_prompt_tokens': maximum, 'grading_file_sha256': sha(REFERENCE / 'official_answer_dict_test.json'),
        'all_models_same_context': True, 'labels_loaded_only_after_prediction_seal': True})


def wait_gpu():
    while True:
        active = []
        for name in ('soft_vote_pilot_20261121', 'mmmu_pro_final_20261121'):
            path = PROJECT / 'logs' / name / 'launch.json'
            if not path.is_file():
                continue
            launch = read(path)
            command = Path('/proc') / str(launch['pid']) / 'cmdline'
            if command.is_file() and (str(PROJECT / 'logs' / name).encode() in command.read_bytes()):
                active.append(name)
        used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if not active and used < 2000:
            return
        status('기존 GPU 작업 종료 대기 중', 실행작업=active, GPU메모리_MiB=used)
        print('기존 GPU 작업 종료를 기다립니다:', active, used, 'MiB', flush=True)
        time.sleep(30)


def chunks(rows):
    for subject in sorted({row['subject'] for row in rows}):
        selected = [row for row in rows if row['subject'] == subject]
        for start in range(0, len(selected), BATCH):
            yield subject + f'_{start:04d}', selected[start:start + BATCH]


def worker(name):
    helper.configure()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    execution = read(JOB / 'execution_lock.json')
    assert execution['code'] == code_lock()
    rows = inputs()
    folder = RESULTS / 'blind_predictions' / name
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    completed = 0
    for key, selected in chunks(rows):
        path = folder / (key + '.json')
        if path.exists():
            previous = read(path)
            assert len(previous) == len(selected)
            assert all(a['id'] == b['id'] and a['input_binding'] == b['input_text_sha256'] for a, b in zip(previous, selected))
            completed += len(previous)
        else:
            pending.append((key, selected))
    if not pending:
        return
    processor = AutoProcessor.from_pretrained(helper.SETTINGS, local_files_only=True)
    functions = helper.primary_functions()
    llm = LLM(model=str(MODELS[name]), limit_mm_per_prompt={'image': 8, 'video': 0},
              max_model_len=execution['max_model_len'], gpu_memory_utilization=0.90,
              seed=3407, trust_remote_code=True, enforce_eager=True, max_num_seqs=4)
    parameters = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                                presence_penalty=1.5, max_tokens=512, seed=3407)
    started, initial = time.monotonic(), completed
    for key, selected in pending:
        owners, requests, letters, flags = [], [], [], []
        try:
            for row in selected:
                value, letter, is_mcq, owned = helper.request(row, processor, functions)
                requests.append(value)
                letters.append(letter)
                flags.append(is_mcq)
                owners.extend(owned)
            generated = llm.generate(requests, parameters, use_tqdm=False)
            records = []
            for row, result, letter, is_mcq in zip(selected, generated, letters, flags):
                output = result.outputs[0]
                predicted = functions['extract_mcq_answer'](output.text, letter) if is_mcq else functions['extract_open_answer'](output.text)
                records.append({'id': row['id'], 'subject': row['subject'], 'is_mcq': is_mcq,
                                'predicted': predicted, 'raw_output': output.text, 'first_raw_output': output.text,
                                'retried': False, 'finish_reason': output.finish_reason, 'input_binding': row['input_text_sha256']})
            retry = [i for i, row in enumerate(records) if row['is_mcq'] and row['predicted'] is None]
            retry_inputs = []
            for i in retry:
                value, _, _, owned = helper.request(selected[i], processor, functions, True)
                retry_inputs.append(value)
                owners.extend(owned)
            if retry:
                for i, result in zip(retry, llm.generate(retry_inputs, parameters, use_tqdm=False)):
                    output = result.outputs[0]
                    records[i].update(predicted=functions['extract_mcq_answer'](output.text, letters[i]),
                                      raw_output=output.text, retried=True, finish_reason=output.finish_reason)
            assert len(records) == len(selected)
            save(folder / (key + '.json'), records)
        finally:
            helper.close_images(owners)
        completed += len(selected)
        rate = (time.monotonic() - started) / max(1, completed - initial)
        status('전체 Test 모델 추론 중', 모델=name, 완료=completed, 전체=TOTAL,
               현재모델_예상남은분=round((TOTAL - completed) * rate / 60, 1), 로그=str(JOB / (name + '.log')))
        print(name, ':', completed, '/', TOTAL, '예측 저장. 채점하지 않았습니다.', flush=True)


def collect(name, rows):
    result = []
    for key, selected in chunks(rows):
        values = read(RESULTS / 'blind_predictions' / name / (key + '.json'))
        assert len(values) == len(selected)
        assert all(a['id'] == b['id'] and a['input_binding'] == b['input_text_sha256'] for a, b in zip(values, selected))
        assert all(not {'answer', 'correct', 'ground_truth', 'gold'} & set(row) for row in values)
        result.extend(values)
    return result


def score(rows):
    execution = read(JOB / 'execution_lock.json')
    assert code_lock() == execution['code']
    assert sha(DATA / 'blind_inputs.json') == execution['input_sha256']
    predictions, sealed = {}, {}
    ids = [row['id'] for row in rows]
    for name in MODELS:
        values = collect(name, rows)
        path = RESULTS / 'blind_predictions' / name / 'all_predictions.json'
        save(path, values)
        sealed[path.relative_to(PROJECT).as_posix()] = sha(path)
        predictions[name] = {row['id']: row for row in values}
    for group, names in GROUPS.items():
        values = []
        for candidate in ids:
            answer, selected = choose_answer({name: predictions[name][candidate]['predicted'] for name in names}, group)
            values.append({'id': candidate, 'predicted': answer, 'selected_model': selected})
        path = RESULTS / 'blind_votes' / (group + '.json')
        save(path, values)
        sealed[path.relative_to(PROJECT).as_posix()] = sha(path)
        predictions[group] = {row['id']: row for row in values}
    fixed(RESULTS / 'prediction_seal.json', {'files': sealed, 'input_sha256': execution['input_sha256']})
    assert sha(REFERENCE / 'official_answer_dict_test.json') == execution['grading_file_sha256']
    answers = read(REFERENCE / 'official_answer_dict_test.json')
    seen = set(read(DATA / 'previously_seen_test_ids.json'))
    grader = helper.primary_functions(grading=True)['grade_open']
    report = {}
    for name, values in predictions.items():
        scored = []
        for row in rows:
            reference = answers[row['id']]
            answer = values[row['id']]['predicted']
            gold = reference['ground_truth']
            assert row['question_type'] == reference['question_type']
            correct = answer in (gold if isinstance(gold, list) else [gold]) if row['question_type'] == 'multiple-choice' else grader(gold, answer)
            scored.append({'id': row['id'], 'subject': row['subject'], 'predicted': answer,
                           'gold': gold, 'correct': bool(correct), 'previously_seen_id': row['id'] in seen})
        save(RESULTS / 'scored' / (name + '.json'), scored)
        metrics = {}
        for subset in ('full_test', 'previously_seen_ids', 'remaining_ids'):
            chosen = [row for row in scored if subset == 'full_test' or row['previously_seen_id'] == (subset == 'previously_seen_ids')]
            total = len(chosen)
            correct = sum(row['correct'] for row in chosen)
            counts = Counter(row['subject'] for row in chosen)
            right = Counter(row['subject'] for row in chosen if row['correct'])
            metrics[subset] = {'correct': correct, 'total': total, 'accuracy': correct / total if total else None,
                               'subject_macro_accuracy': sum(right[subject] / count for subject, count in counts.items()) / len(counts) if counts else None,
                               'subjects': {subject: {'correct': right[subject], 'total': count} for subject, count in sorted(counts.items())}}
        report[name] = metrics
    save(RESULTS / 'comparison.json', {'results': report, 'execution': execution,
         'limitations': 'Custom fixed MMMU grading, not an official leaderboard score. Full Test includes previously observed IDs. Remaining IDs have not undergone a complete image/text training overlap audit. Multiple candidates and a single training seed limit superiority claims. No probability-ensemble tuning uses these results.'})
    status('전체 Test 평가 완료', 결과={name: value['full_test']['correct'] for name, value in report.items()}, 결과파일=str(RESULTS / 'comparison.json'))


def run():
    helper.configure()
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('전체 Test 작업이 이미 실행 중입니다.')
        try:
            for package, version in (('torch', '2.8.0'), ('transformers', '4.57.6'), ('vllm', '0.11.0')):
                assert importlib.metadata.version(package).split('+')[0] == version
            rows = prepare()
            if not (JOB / 'execution_lock.json').exists():
                preflight(rows)
            execution = read(JOB / 'execution_lock.json')
            assert execution['code'] == code_lock()
            assert execution['input_sha256'] == sha(DATA / 'blind_inputs.json')
            assert execution['protocol_sha256'] == sha(DATA / 'protocol.json')
            previous = read(PROJECT / 'logs/independent_900_20261121/model_binary_lock.json')
            model_lock = {}
            for name, folder in MODELS.items():
                assert folder == Path(previous[name]['path'])
                assert sha(folder / 'config.json') == previous[name]['config_sha256']
                for filename, expected in previous[name]['weights'].items():
                    status('고정 모델 파일 검사 중', 모델=name, 파일=filename)
                    assert sha(folder / filename) == expected
                model_lock[name] = previous[name]
            fixed(JOB / 'model_binary_lock.json', model_lock)
            for name in MODELS:
                if all((RESULTS / 'blind_predictions' / name / (key + '.json')).exists() for key, _ in chunks(rows)):
                    collect(name, rows)
                    continue
                wait_gpu()
                logpath = JOB / (name + '.log')
                status('전체 Test 모델 로딩 중', 모델=name, 로그=str(logpath))
                with logpath.open('a', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name],
                                   cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            status('전체 예측 완료. 고정 투표와 채점 중')
            score(rows)
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise


def launch():
    helper.configure()
    JOB.mkdir(parents=True, exist_ok=True)
    path = JOB / 'run.py'
    source = Path(__file__).read_bytes()
    if path.exists():
        assert path.read_bytes() == source, '기존 실행 파일과 다릅니다.'
    else:
        path.write_bytes(source)
    with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(path), '--run'], cwd=PROJECT,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(JOB / 'launch.json', {'pid': process.pid, 'script': str(path)})
    print('전체 Test 백그라운드 실행 요청 완료')
    print('기존 GPU 작업이 있으면 입력 준비 후 기다립니다.')
    print('프로세스 번호:', process.pid)
    print('로그:', JOB / 'runner.log')


def monitor(watch):
    while True:
        state = read(JOB / 'status.json') if (JOB / 'status.json').exists() else {'상태': '아직 상태 파일 없음'}
        print('\n확인 시각:', datetime.now().isoformat(timespec='seconds'))
        print(json.dumps(state, ensure_ascii=False, indent=2))
        for name in MODELS:
            folder = RESULTS / 'blind_predictions' / name
            count = sum(len(read(path)) for path in folder.glob('*.json') if path.name != 'all_predictions.json') if folder.exists() else 0
            print(name, ':', count, '/', TOTAL, '개 저장')
        info = JOB / 'launch.json'
        if info.exists():
            pid = read(info)['pid']
            command = Path('/proc') / str(pid) / 'cmdline'
            alive = command.exists() and str(JOB / 'run.py').encode() in command.read_bytes()
            print('순차 실행 프로세스:', pid, '실행 중' if alive else '실행되지 않음')
        log = Path(state.get('로그', str(JOB / 'runner.log')))
        if log.exists():
            print('최근 로그:', log)
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').replace('\r', '\n').splitlines()[-8:]))
        if not watch or state['상태'] in TERMINAL:
            break
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', choices=list(MODELS))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.run:
        run()
    elif args.worker:
        worker(args.worker)
    else:
        try:
            monitor(args.watch)
        except KeyboardInterrupt:
            print('\n확인 화면만 종료했습니다. 백그라운드 평가는 계속됩니다.')
