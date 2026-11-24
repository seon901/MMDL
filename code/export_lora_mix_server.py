from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import tarfile
ROOT = Path('/shared_dir/mmdl')
JOB = ROOT / 'checkpoints/lora_mix_20261121'
EXPORTS = ROOT / 'exports'
MODELS = {'mix_90_10': JOB / 'adapters/mix_90_10'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
assert read(JOB / 'status.json')['상태'] == '세 혼합 비율의 개발 평가 완료'
assert (JOB / 'mix_comparison.json').is_file()
members = {}
models = {}
for label, folder in MODELS.items():
    config = read(folder / 'adapter_config.json')
    assert config['r'] == 16 and config['lora_alpha'] == 16
    prefix = f'candidates/{label}/adapter'
    for name in ['adapter_config.json', 'adapter_model.safetensors']:
        path = folder / name
        assert path.is_file() and path.stat().st_size > 0
        members[f'{prefix}/{name}'] = path
    models[label] = {'adapter_path': prefix, 'rank': 16, 'alpha': 16, 'source': str(folder), 'training': False, 'mix_weights': [0.9, 0.1]}
    for path in folder.iterdir():
        if path.is_file() and path.suffix in {'.json', '.txt', '.log', '.md'}:
            members[f'records/{label}/{path.name}'] = path
for tag, run in [('mix_run', JOB)]:
    for path in run.iterdir():
        if path.is_file() and path.suffix in {'.py', '.json', '.txt', '.log'}:
            members[f'records/{tag}/{path.name}'] = path
for name in ['evaluate_lora_mix_server.py', 'train_verified_hard_server.py', 'r20_strength_server.py', 'mine_verified_hard_server.py']:
    path = ROOT / name
    assert path.is_file()
    members[f'records/scripts/{name}'] = path
manifest = {'purpose': 'Fixed MMMU evaluation of the internally selected 90:10 R20 and reasoning weighted-delta model', 'models': models, 'expected_local_base_model': 'Qwen/Qwen3-VL-4B-Instruct', 'expected_local_base_revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17', 'locked_eval_script_sha256': '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8', 'dataset_revision': '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68', 'design': read(JOB / 'experiment_plan.json'), 'scope': 'Adapters and reproducibility records. Base weights, images, optimizer states and training environments are excluded.', 'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in members.items()}}
EXPORTS.mkdir(exist_ok=True)
archive = EXPORTS / ('mmmu_mix_90_10_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.tar.gz')
temporary = archive.with_name(archive.name + '.partial')
with tarfile.open(temporary, 'x:gz', compresslevel=3) as tar:
    for name, path in members.items():
        tar.add(path, arcname=name, recursive=False)
    content = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    entry = tarfile.TarInfo('manifest.json')
    entry.size = len(content)
    tar.addfile(entry, io.BytesIO(content))
temporary.rename(archive)
with tarfile.open(archive, 'r:gz') as tar:
    assert len(tar.getnames()) == len(members) + 1
    for name, item in manifest['files'].items():
        stream = tar.extractfile(name)
        assert stream is not None and hashlib.sha256(stream.read()).hexdigest() == item['sha256'], f'내부 검증 실패: {name}'
checksum = sha(archive)
sidecar = archive.with_name(archive.name + '.sha256')
sidecar.write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
print('=== 90대10 혼합 평가 묶음 생성 완료 ===')
print('학교 호스트 압축 파일:', '/home/gpu/shared_dir/mmdl/exports/' + archive.name)
print('학교 호스트 검증 파일:', '/home/gpu/shared_dir/mmdl/exports/' + sidecar.name)
print('크기 MiB:', round(archive.stat().st_size / 1024 ** 2, 2))
print('SHA256:', checksum)
print('내부 파일 검증: 통과')
print('개인 컴퓨터 전송은 아직 수행하지 않았습니다.')
