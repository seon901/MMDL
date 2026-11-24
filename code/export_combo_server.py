from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import tarfile

ROOT = Path('/shared_dir/mmdl')
JOB = ROOT / 'checkpoints/replay_reasoning_compare_20261120_212449_530854'
EXPORTS = ROOT / 'exports'
LABELS = ['replay_native_reasoning', 'replay_answer_control']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

assert read(JOB / 'overall_status.json')['상태'] == '조합 학습과 이미지 및 텍스트 평가 완료'
assert (JOB / 'factorial_comparison.json').is_file()
assert (JOB / 'combo_comparison.json').is_file()
members = {}
models = {}
for label in LABELS:
    folder = JOB / label
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
        if path.is_file() and path.suffix in {'.json', '.txt', '.log', '.md'}:
            members[f'records/{label}/{path.name}'] = path
for path in JOB.iterdir():
    if path.is_file() and path.suffix in {'.py', '.json', '.txt', '.log'}:
        members[f'records/combo_run/{path.name}'] = path
plan = read(JOB / 'combo_plan.json')
data = Path(read(JOB / 'execution_plan.json')['데이터'])
for label in LABELS:
    for name in ['train_provenance.json', 'internal_dev_provenance.json']:
        path = data / label / name
        assert path.is_file()
        members[f'records/provenance/{label}/{name}'] = path
preparer = ROOT / 'code/experiment_scripts/prepare_replay_reasoning_v1.py'
if preparer.is_file():
    members['records/prepare_replay_reasoning_v1.py'] = preparer
manifest = {
    'purpose': 'Fixed MMMU evaluation of the replay and reasoning combination',
    'models': models,
    'expected_local_base_model': 'Qwen/Qwen3-VL-4B-Instruct',
    'expected_local_base_revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
    'locked_eval_script_sha256': '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8',
    'dataset_revision': '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',
    'design': plan,
    'scope': 'Adapters and records. Base weights and full image datasets are not included.',
    'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in members.items()},
}
EXPORTS.mkdir(exist_ok=True)
archive = EXPORTS / ('mmmu_combo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.tar.gz')
temporary = archive.with_name(archive.name + '.partial')
print('=== 조합 평가용 파일 압축 중 ===', flush=True)
with tarfile.open(temporary, 'x:gz', compresslevel=3) as tar:
    for name, path in members.items():
        tar.add(path, arcname=name, recursive=False)
    content = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    info = tarfile.TarInfo('manifest.json')
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))
temporary.rename(archive)
with tarfile.open(archive, 'r:gz') as tar:
    assert len(tar.getnames()) == len(members) + 1
    for name, item in manifest['files'].items():
        handle = tar.extractfile(name)
        assert handle is not None
        assert hashlib.sha256(handle.read()).hexdigest() == item['sha256'], f'내부 파일 검증 실패: {name}'
checksum = sha(archive)
sidecar = archive.with_name(archive.name + '.sha256')
sidecar.write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
print('=== 조합 평가용 묶음 생성 완료 ===')
print('학교 호스트 압축 파일:', '/home/gpu/shared_dir/mmdl/exports/' + archive.name)
print('학교 호스트 검증 파일:', '/home/gpu/shared_dir/mmdl/exports/' + sidecar.name)
print('크기 MiB:', round(archive.stat().st_size / 1024**2, 2))
print('SHA256:', checksum)
print('내부 파일 검증: 통과')
print('개인 컴퓨터 전송은 아직 수행하지 않았습니다.')