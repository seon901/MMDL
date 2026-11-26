from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from io import BytesIO
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
JOB = PROJECT / 'logs/mmmu_pro_base_R20_20261121'
RESULTS = PROJECT / 'results/mmmu_pro_base_R20_20261121'
REVISION = '0d7426df4ccb3d8704a992fe6850abee7c262127'
REFERENCE = PROJECT / 'evaluation_reference/MMMU_Pro'
DATA = REFERENCE / REVISION
MODELS = {
    'base': Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'),
    'R20': PROJECT / 'merged_models/mmmu_candidates_20261120_004215_155629/R20',
}
CONFIGS = {'standard': 'standard (10 options)', 'vision': 'vision'}
INSTRUCTION = 'Choose the correct option. Respond with only one line: Answer: X, replacing X with its option letter.'
PROTOCOL = {
    'name': 'MMMU-Pro local direct evaluation v1', 'revision': REVISION,
    'settings': CONFIGS, 'expected_rows_each': 1730,
    'instruction': INSTRUCTION, 'max_new_tokens': 512,
    'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
    'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'seed': 3407,
    'min_pixels': 200704, 'max_pixels': 802816,
    'max_model_len': 16384, 'max_num_seqs': 1, 'max_num_batched_tokens': 2048,
    'standard_images': 'Each referenced image is shown once with a Figure label. Question and option references use the same labels.',
    'vision_input': 'Screenshot and shared instruction only. Option text and gold answers are never included separately in the model input.',
    'grading': 'Last explicit Answer line containing one valid letter, or a whole response containing only one valid letter. No random fallback and no retry.',
    'limitations': 'Separate local protocol, not an official leaderboard score. Training overlap with the full MMMU-Pro set has not been exhaustively excluded. No selection of models or prompts using these scores.',
}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def options(row):
    values = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
    assert isinstance(values, list) and 2 <= len(values) <= 26
    assert all(isinstance(x, str) for x in values)
    return values

def input_spec(row, setting):
    if setting == 'vision':
        assert row.get('image') is not None
        return [('image', 'image'), ('text', INSTRUCTION)]
    choices = options(row)
    text = 'Question: ' + row['question'] + '\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(choices))
    references = list(dict.fromkeys(re.findall(r'<image\s+(\d+)>', text)))
    assert all(1 <= int(n) <= 7 and row.get('image_' + n) is not None for n in references)
    converted = re.sub(r'<image\s+(\d+)>', lambda m: 'Figure ' + m.group(1), text)
    spec = []
    for n in references:
        spec.extend([('text', 'Figure ' + n + ':'), ('image', 'image_' + n)])
    spec.append(('text', converted + '\n' + INSTRUCTION))
    return spec

def load_image(cell):
    from PIL import Image
    assert isinstance(cell, dict) and cell.get('bytes'), '평가 이미지의 내장 데이터를 찾지 못했습니다.'
    with Image.open(BytesIO(cell['bytes'])) as opened:
        return opened.convert('RGB')

def input_hash(row, setting):
    h = hashlib.sha256()
    for kind, value in input_spec(row, setting):
        h.update(kind.encode() + b'\0')
        h.update(value.encode('utf-8') if kind == 'text' else hashlib.sha256(row[value]['bytes']).digest())
        h.update(b'\0')
    return h.hexdigest()

def iter_rows(setting):
    import pyarrow.parquet as pq
    for path in sorted((DATA / CONFIGS[setting]).glob('*.parquet')):
        columns = ['id', 'options', 'answer', 'subject'] + (['image'] if setting == 'vision' else ['question'] + [f'image_{i}' for i in range(1, 8)])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=columns):
            yield from batch.to_pylist()

def parse_answer(text, count):
    matches = re.findall(r'^\s*Answer\s*:\s*\(?([A-Z])\)?[.\s]*$', text, flags=re.I | re.M)
    value = matches[-1].upper() if matches else None
    if value is None:
        match = re.fullmatch(r'\s*\(?([A-Z])\)?[.\s]*', text, flags=re.I)
        value = match.group(1).upper() if match else None
    return value if value and ord(value) - 65 < count else None

def prepare():
    import PIL
    manifest = read(REFERENCE / 'reference_manifest.json')
    assert manifest['revision'] == REVISION
    for name, record in manifest['files'].items():
        path = DATA / name
        assert path.is_file() and path.stat().st_size == record['bytes']
        assert digest(path) == record['sha256'], f'원본 검증 실패: {path}'
    bindings = {}
    for setting in CONFIGS:
        entries = []
        seen = set()
        for row in iter_rows(setting):
            assert row['id'] not in seen
            seen.add(row['id'])
            choices = options(row)
            assert row['answer'] in [chr(65+i) for i in range(len(choices))]
            images = [value for kind, value in input_spec(row, setting) if kind == 'image']
            assert len(images) <= 7
            for field in images:
                image = load_image(row[field])
                image.close()
            entries.append({'id': row['id'], 'subject': row['subject'], 'gold': row['answer'], 'option_count': len(choices), 'input_sha256': input_hash(row, setting), 'images': len(images)})
        assert len(entries) == 1730
        bindings[setting] = entries
        save(JOB / (setting + '_inputs.json'), entries)
        print(f'{setting}: 입력과 이미지 {len(entries)}개 검사 통과', flush=True)
    assert [r['id'] for r in bindings['standard']] == [r['id'] for r in bindings['vision']]
    assert all(a['gold'] == b['gold'] and a['option_count'] == b['option_count'] for a, b in zip(bindings['standard'], bindings['vision']))
    protocol = {**PROTOCOL, 'evaluator_sha256': digest(Path(__file__)), 'reference_manifest_sha256': digest(REFERENCE / 'reference_manifest.json'), 'model_paths': {k: str(v) for k, v in MODELS.items()}, 'R20_merge_record_sha256': digest(MODELS['R20'] / 'merge_record.json')}
    path = JOB / 'protocol.json'
    if path.exists():
        assert read(path) == protocol, '이전 실행과 평가 조건이 다릅니다. 기존 결과에 이어 쓰지 않습니다.'
    else:
        save(path, protocol)
    return protocol

def summary(values):
    groups = defaultdict(list)
    for row in values:
        groups[row['subject']].append(row)
    return {'total': len(values), 'correct': sum(r['correct'] for r in values), 'accuracy': sum(r['correct'] for r in values) / len(values), 'parse_failures': sum(r['predicted'] is None for r in values), 'length_limit': sum(r['finish_reason'] == 'length' for r in values), 'subjects': {name: {'correct': sum(r['correct'] for r in rows), 'total': len(rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows)} for name, rows in groups.items()}}

def worker(model_name, setting):
    from resume_mmmu_candidates import configure
    configure()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    path = RESULTS / model_name / (setting + '_predictions.json')
    values = read(path) if path.exists() else []
    bindings = read(JOB / (setting + '_inputs.json'))
    assert len(values) <= len(bindings)
    for old, binding in zip(values, bindings):
        assert all(old[k] == binding[k] for k in ['id', 'subject', 'gold', 'option_count', 'input_sha256'])
        assert old['predicted'] == parse_answer(old['raw'], old['option_count'])
        assert old['correct'] == (old['predicted'] == old['gold'])
    if len(values) == 1730:
        save(path.parent / (setting + '_summary.json'), summary(values))
        return
    processor = AutoProcessor.from_pretrained(MODELS['R20'], local_files_only=True)
    llm = LLM(model=str(MODELS[model_name]), dtype='bfloat16', trust_remote_code=False,
              limit_mm_per_prompt={'image': 7, 'video': 0}, max_model_len=16384,
              gpu_memory_utilization=0.90, seed=3407, enforce_eager=True,
              max_num_seqs=1, max_num_batched_tokens=2048, enable_chunked_prefill=True,
              mm_processor_kwargs={'min_pixels': 200704, 'max_pixels': 802816})
    sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                              presence_penalty=0.0, max_tokens=512, seed=3407)
    started = time.monotonic()
    initial = len(values)
    for index, row in enumerate(iter_rows(setting)):
        binding = bindings[index]
        assert input_hash(row, setting) == binding['input_sha256']
        if index < initial:
            continue
        content = []
        for kind, value in input_spec(row, setting):
            if kind == 'text':
                content.append({'type': 'text', 'text': value})
            else:
                content.append({'type': 'image', 'image': load_image(row[value]), 'min_pixels': 200704, 'max_pixels': 802816})
        messages = [{'role': 'user', 'content': content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        assert not videos
        request = {'prompt': prompt}
        if images:
            request['multi_modal_data'] = {'image': images}
        result = llm.generate([request], sampling, use_tqdm=False)[0]
        output = result.outputs[0]
        assert len(result.prompt_token_ids) + 512 <= 16384, '입력이 고정 문맥 한도를 초과했습니다.'
        predicted = parse_answer(output.text, binding['option_count'])
        values.append({**binding, 'predicted': predicted, 'correct': predicted == binding['gold'], 'raw': output.text,
                       'finish_reason': output.finish_reason, 'prompt_tokens': len(result.prompt_token_ids), 'output_tokens': len(output.token_ids)})
        for item in content:
            if item['type'] == 'image':
                item['image'].close()
        if images:
            for image in images:
                image.close()
        if len(values) % 5 == 0 or len(values) == 1730:
            save(path, values)
            elapsed = time.monotonic() - started
            done = len(values) - initial
            remaining = elapsed / done * (1730 - len(values))
            status('평가 중', 모델=model_name, 구성=setting, 완료=len(values), 전체=1730,
                   현재구성_예상남은초=round(remaining), 로그=str(JOB / f'{model_name}_{setting}.log'))
            print(f'{model_name} {setting}: {len(values)}/1730, 정답 {sum(r["correct"] for r in values)}, 예상 남은 시간 {remaining/60:.1f}분', flush=True)
    save(path.parent / (setting + '_summary.json'), summary(values))

def compare():
    report = {}
    for setting in CONFIGS:
        base = read(RESULTS / 'base' / (setting + '_predictions.json'))
        trained = read(RESULTS / 'R20' / (setting + '_predictions.json'))
        assert len(base) == len(trained) == 1730
        assert all(all(a[k] == b[k] for k in ['id', 'subject', 'gold', 'option_count', 'input_sha256']) for a, b in zip(base, trained))
        recovered = sum(not a['correct'] and b['correct'] for a, b in zip(base, trained))
        regressed = sum(a['correct'] and not b['correct'] for a, b in zip(base, trained))
        n = recovered + regressed
        pvalue = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(recovered, regressed) + 1)) / (2 ** n)) if n else 1.0
        report[setting] = {'base': summary(base), 'R20': summary(trained), '회복': recovered, '퇴보': regressed, '정답 순증가': recovered-regressed, 'McNemar exact p': pvalue}
    save(RESULTS / 'comparison.json', {'결과': report, 'protocol': read(JOB / 'protocol.json'), '주의': PROTOCOL['limitations']})
    status('완료', 결과={setting: {name: report[setting][name]['correct'] for name in MODELS} for setting in CONFIGS}, 결과파일=str(RESULTS / 'comparison.json'))
    print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)

def run():
    from resume_mmmu_candidates import configure
    configure()
    with (JOB / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('같은 MMMU-Pro 작업이 이미 실행 중입니다.')
        try:
            status('원본과 입력 사전 검사 중')
            assert digest(PROJECT / 'scripts/eval_mmmu_vllm.py') == '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
            for name, folder in MODELS.items():
                assert (folder / 'config.json').is_file(), f'모델 폴더 확인 필요: {folder}'
                assert list(folder.glob('model*.safetensors')), f'모델 가중치 없음: {folder}'
            assert importlib.metadata.version('vllm') == '0.11.0'
            merge = read(MODELS['R20'] / 'merge_record.json')
            assert merge['모델'] == 'R20' and merge['적용 배율'] == 1.0
            assert merge['어댑터 SHA256'] == '4f23f6ecd950630fe9d6208b42550e166242a790d37aaa6fe50bb5975df49784'
            for filename, expected in merge['처리 설정 SHA256'].items():
                assert digest(MODELS['R20'] / filename) == expected, f'처리 설정 변경: {filename}'
            prepare()
            save(JOB / 'environment.json', {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'vllm', 'qwen-vl-utils', 'safetensors']})
            for setting in CONFIGS:
                for name in MODELS:
                    logpath = JOB / f'{name}_{setting}.log'
                    status('모델 로딩 또는 평가 중', 모델=name, 구성=setting, 로그=str(logpath))
                    with logpath.open('a', encoding='utf-8') as log:
                        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name, setting], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            compare()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise

def launch():
    JOB.mkdir(parents=True, exist_ok=True)
    from resume_mmmu_candidates import configure
    configure()
    with (JOB / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('같은 MMMU-Pro 작업이 이미 실행 중입니다.')
    with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run'], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(JOB / 'launch.json', {'pid': process.pid, '시각': datetime.now().isoformat()})
    print('MMMU-Pro 백그라운드 실행 요청 완료')
    print('프로세스 번호:', process.pid)
    print('상태 파일:', JOB / 'status.json')
    print('로그:', JOB / 'runner.log')

def monitor(watch):
    while True:
        print('\n=== MMMU-Pro 진행 상황 ===')
        print('확인 시각:', datetime.now().isoformat(timespec='seconds'))
        state = read(JOB / 'status.json') if (JOB / 'status.json').exists() else {'상태': '아직 상태 파일 없음'}
        print(json.dumps(state, ensure_ascii=False, indent=2))
        for setting in CONFIGS:
            for name in MODELS:
                path = RESULTS / name / (setting + '_predictions.json')
                rows = read(path) if path.exists() else []
                print(f'{name} {setting}: {len(rows)}/1730 저장 완료')
        logpath = Path(state.get('로그', str(JOB / 'runner.log')))
        if logpath.is_file():
            print('최근 로그:', logpath)
            print('마지막 로그 갱신:', round(time.time()-logpath.stat().st_mtime), '초 전')
            print('\n'.join(logpath.read_text(encoding='utf-8', errors='replace').replace('\r', '\n').splitlines()[-8:]))
        if not watch or state['상태'] in ['완료', '오류로 중단']:
            break
        time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('MODEL', 'SETTING'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.run:
        run()
    elif args.worker:
        worker(*args.worker)
    else:
        try:
            monitor(args.watch)
        except KeyboardInterrupt:
            print('\n확인 화면만 종료했습니다. 백그라운드 평가는 계속됩니다.')
