from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import tarfile

ROOT = Path('/shared_dir/mmdl')
ANSWER = ROOT / 'checkpoints/verified_hard_compare_20261121'
REASONING = ROOT / 'checkpoints/verified_hard_reasoning_20261121'
EXPORTS = ROOT / 'exports'
MODELS = {'hard_answer': ANSWER / 'hard_answer', 'hard_reasoning': REASONING / 'hard_reasoning'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()

assert read(REASONING / 'status.json')['상태'] == '정답형과 풀이형 개발 비교 완료'
assert (REASONING / 'answer_reasoning_comparison.json').is_file()
members = {}
models = {}
for label, folder in MODELS.items():
    assert read(folder / 'trainer_state.json')['global_step'] == 500
    config = read(folder / 'adapter_config.json')
    assert config['r'] == 8 and config['lora_alpha'] == 16
    prefix = f'candidates/{label}/adapter'
    for name in ['adapter_config.json', 'adapter_model.safetensors']:
        path = folder / name
        assert path.is_file() and path.stat().st_size > 0
        members[f'{prefix}/{name}'] = path
    models[label] = {'adapter_path': prefix, 'steps': 500, 'rank': 8, 'alpha': 16, 'source': str(folder)}
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {'.json','.txt','.log','.md'}:
            members[f'records/{label}/{path.name}'] = path
for tag, run in [('answer_run', ANSWER), ('reasoning_run', REASONING)]:
    for path in run.iterdir():
        if path.is_file() and path.suffix in {'.py','.json','.txt','.log'}:
            members[f'records/{tag}/{path.name}'] = path
for label, parent in [('hard_answer', ANSWER), ('hard_reasoning', REASONING)]:
    for name in ['train.json', 'internal_dev.json', 'selection.json']:
        path = parent / 'data' / label / name
        assert path.is_file()
        members[f'records/data/{label}/{name}'] = path
for name in ['mine_verified_hard_server.py','train_verified_hard_server.py','train_verified_reasoning_server.py','r20_strength_server.py']:
    path = ROOT / name
    assert path.is_file()
    members[f'records/scripts/{name}'] = path
for name in ['profile_plan.json','profile_summary.json','predictions.json','ranked_candidates.json']:
    path = ROOT / 'checkpoints/verified_hard_profile_20261121' / name
    assert path.is_file()
    members[f'records/mining/{name}'] = path
manifest = {
    'purpose': 'Fixed MMMU comparison of matched hard-example answer-only and concise-reasoning training',
    'models': models,
    'expected_local_base_model': 'Qwen/Qwen3-VL-4B-Instruct',
    'expected_local_base_revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    'locked_eval_script_sha256': '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8',
    'dataset_revision': '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',
    'design': read(REASONING / 'experiment_plan.json'),
    'scope': 'Adapters and reproducibility records. Base weights, images, optimizer states and training environments are excluded.',
    'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name,path in members.items()},
}
EXPORTS.mkdir(exist_ok=True)
archive = EXPORTS / ('mmmu_hard_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.tar.gz')
temporary = archive.with_name(archive.name + '.partial')
with tarfile.open(temporary, 'x:gz', compresslevel=3) as tar:
    for name,path in members.items():
        tar.add(path, arcname=name, recursive=False)
    content = json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    entry = tarfile.TarInfo('manifest.json')
    entry.size = len(content)
    tar.addfile(entry,io.BytesIO(content))
temporary.rename(archive)
with tarfile.open(archive,'r:gz') as tar:
    assert len(tar.getnames()) == len(members)+1
    for name,item in manifest['files'].items():
        stream=tar.extractfile(name)
        assert stream is not None and hashlib.sha256(stream.read()).hexdigest()==item['sha256'], f'내부 검증 실패: {name}'
checksum = sha(archive)
sidecar = archive.with_name(archive.name+'.sha256')
sidecar.write_text(f'{checksum}  {archive.name}\n',encoding='utf-8')
print('=== 정답형과 풀이형 평가 묶음 생성 완료 ===')
print('학교 호스트 압축 파일:', '/home/gpu/shared_dir/mmdl/exports/'+archive.name)
print('학교 호스트 검증 파일:', '/home/gpu/shared_dir/mmdl/exports/'+sidecar.name)
print('크기 MiB:', round(archive.stat().st_size/1024**2,2))
print('SHA256:',checksum)
print('내부 파일 검증: 통과')
print('개인 컴퓨터 전송은 아직 수행하지 않았습니다.')
