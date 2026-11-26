from pathlib import Path
from datetime import datetime
from collections import Counter
import ast
import hashlib
import json
import re

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
ROOT = PROJECT / 'evaluation_reference/MMMU_Pro'
CONFIGS = ['standard (10 options)', 'vision']

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

assert PROJECT.is_dir(), '개인 컴퓨터 WSL에서 실행해 주세요.'
ROOT.mkdir(parents=True, exist_ok=True)
lock = ROOT / 'revision_lock.json'
if lock.is_file():
    revision = json.loads(lock.read_text(encoding='utf-8'))['revision']
else:
    revision = HfApi().dataset_info('MMMU/MMMU_Pro').sha
    assert re.fullmatch(r'[0-9a-f]{40}', revision)
    save(lock, {'repository': 'MMMU/MMMU_Pro', 'revision': revision, 'fixed_at': datetime.now().isoformat(), 'purpose': 'evaluation_only'})
folder = ROOT / revision
print('=== MMMU-Pro 평가 원본 다운로드 ===', flush=True)
print('고정 revision:', revision, flush=True)
print('저장 폴더:', folder, flush=True)
complete = True
for config in CONFIGS:
    existing = sorted((folder / config).glob('test-*.parquet'))
    try:
        complete = complete and bool(existing) and sum(pq.ParquetFile(path).metadata.num_rows for path in existing) == 1730
    except Exception:
        complete = False
if complete:
    print('기존 파일 6개를 재사용합니다. 추가 다운로드 없이 검사합니다.', flush=True)
else:
    snapshot_download(repo_id='MMMU/MMMU_Pro', repo_type='dataset', revision=revision, local_dir=folder, allow_patterns=['README.md', 'standard (10 options)/test-*.parquet', 'vision/test-*.parquet'], max_workers=2)
result = {}
all_files = {}
for config in CONFIGS:
    files = sorted((folder / config).glob('test-*.parquet'))
    assert files, f'평가 파일이 없습니다: {config}'
    total = 0
    subjects = Counter()
    option_counts = Counter()
    seen = set()
    max_references = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        required = {'id', 'subject', 'answer', 'options'} | ({'question', 'image_1'} if config.startswith('standard') else {'image'})
        assert required <= names, f'예상과 다른 데이터 구조: {config}'
        total += parquet.metadata.num_rows
        columns = ['id', 'subject', 'options', 'answer'] + (['question'] if config.startswith('standard') else [])
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                assert row['id'] not in seen
                seen.add(row['id'])
                subjects[row['subject']] += 1
                options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
                assert isinstance(options, list) and 2 <= len(options) <= 26
                gold = row['answer']
                assert isinstance(gold, str) and len(gold) == 1 and 'A' <= gold <= chr(64 + len(options)), '정답과 선택지 연결 오류: ' + str(row['id'])
                option_counts[len(options)] += 1
                if 'question' in row:
                    text = row['question'] + '\n' + '\n'.join(options)
                    max_references = max(max_references, len(re.findall(r'<image\s+\d+>', text)))
        all_files[path.relative_to(folder).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    assert total == len(seen) == 1730, f'문항 수가 예상과 다릅니다: {config}, {total}'
    result[config] = {'문항 수': total, '파일 수': len(files), '선택지 수 집계': dict(option_counts), '최대 이미지 참조 횟수': max_references, '과목별 수량': dict(subjects)}
    print(config, ':', total, '개 준비 완료', flush=True)
report = {'repository': 'MMMU/MMMU_Pro', 'revision': revision, 'folder': str(folder), 'evaluation_only': True, 'configs': result, 'files': all_files, '주의': '평가 원본 준비만 완료했습니다. 모델 추론과 채점은 아직 실행하지 않았습니다.'}
save(ROOT / 'reference_manifest.json', report)
print('=== MMMU-Pro 평가 원본 준비 완료 ===')
print(json.dumps({'revision': revision, '설정별 문항 수': {key: value['문항 수'] for key, value in result.items()}, '표준형 최대 이미지 참조 횟수': result['standard (10 options)']['최대 이미지 참조 횟수'], '기록 파일': str(ROOT / 'reference_manifest.json')}, ensure_ascii=False, indent=2))
