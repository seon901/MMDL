from pathlib import Path, PurePosixPath
from datetime import datetime
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
IDENTIFIER = 'mmmu_candidates_20261120_004215_155629'
JOB = PROJECT / 'logs' / IDENTIFIER
ARCHIVE = PROJECT / 'transfer_packages' / (IDENTIFIER + '.tar.gz')
ARCHIVE_SHA = '4d7c15ac581872ba54a50d08dbf4913311f384399dfbb33c9c6001c8a45472c9'
PACKAGE = PROJECT / 'eval_candidates' / IDENTIFIER
MERGED_ROOT = PROJECT / 'merged_models' / IDENTIFIER
RESULT_ROOT = PROJECT / 'results' / IDENTIFIER
BASE_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
DATA_REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
BASE = Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots') / BASE_REVISION
SETTINGS = PROJECT / 'merged_models/recipe_a_lr_2e-4_step500_20261116_232956_544892'
DEPENDENCIES = PROJECT / 'merge_dependencies_Sb7GdJ'
EVALUATOR = PROJECT / 'scripts/eval_mmmu_vllm.py'
EVALUATOR_SHA = '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
BASE_RESULT = PROJECT / 'results_vllm_v2'
A_RESULT = PROJECT / 'results/recipe_a_lr_2e-4_step500_retry_20261117_004730'
LABELS = ['native_reasoning', 'answer_control', 'R20']
ACTIVE_LOG = JOB / 'runner.log'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def fixed_json(path, value):
    if path.exists():
        assert read(path) == value, f'기존 실행 기록과 다릅니다: {path}'
    else:
        save(path, value)

def validated_members(tar, manifest):
    members = tar.getmembers()
    names = [member.name for member in members]
    assert len(names) == len(set(names)), '압축 내부에 중복 경로가 있습니다.'
    assert set(names) == set(manifest['files']) | {'manifest.json'}
    assert sum(member.size for member in members) < 512 * 1024**2
    for member in members:
        path = PurePosixPath(member.name)
        assert member.isfile(), '일반 파일만 허용합니다.'
        assert not path.is_absolute() and '..' not in path.parts
        assert '\\' not in member.name and ':' not in member.name
    return members

def unpack():
    assert digest(ARCHIVE) == ARCHIVE_SHA, '전송 압축 파일 SHA256 불일치'
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        handle = tar.extractfile('manifest.json')
        assert handle is not None
        manifest = json.load(handle)
        assert set(manifest['models']) == set(LABELS)
        assert manifest['expected_local_base_revision'] == BASE_REVISION
        assert manifest['locked_eval_script_sha256'] == EVALUATOR_SHA
        members = validated_members(tar, manifest)
        if not PACKAGE.exists():
            PACKAGE.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=IDENTIFIER + '_extract_', dir=PACKAGE.parent))
            for member in members:
                destination = staging / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                handle = tar.extractfile(member)
                assert handle is not None
                with destination.open('xb') as target:
                    shutil.copyfileobj(handle, target)
                if member.name != 'manifest.json':
                    assert digest(destination) == manifest['files'][member.name]['sha256']
            staging.rename(PACKAGE)
        assert read(PACKAGE / 'manifest.json') == manifest
        for name, item in manifest['files'].items():
            path = PACKAGE / name
            assert not path.is_symlink()
            assert path.stat().st_size == item['bytes']
            assert digest(path) == item['sha256'], f'내부 파일 변경: {name}'
    return manifest

def results(folder, subjects):
    records = {}
    for subject in subjects:
        values = read(folder / f'{subject}.json')
        assert len(values) == 30, f'과목 문항 수 불일치: {subject}'
        for row in values:
            assert row['subject'] == subject
            assert row['id'] not in records
            records[row['id']] = row
    assert len(records) == 900
    return records

def check():
    assert PROJECT.is_dir() and ARCHIVE.is_file(), '개인 컴퓨터 WSL의 프로젝트와 전송 파일이 필요합니다.'
    assert digest(EVALUATOR) == EVALUATOR_SHA, '기존 MMMU 평가 코드가 변경됐습니다.'
    for package, version in [('torch', '2.8.0'), ('transformers', '4.57.6'), ('vllm', '0.11.0')]:
        assert importlib.metadata.version(package).split('+')[0] == version, f'평가 환경 버전 불일치: {package}'
    assert shutil.which('ninja'), '기존 venv_vllm/bin/ninja를 찾지 못했습니다.'
    assert BASE.is_dir() and DEPENDENCIES.is_dir()
    index = read(BASE / 'model.safetensors.index.json')
    assert all((BASE / name).is_file() for name in set(index['weight_map'].values()))
    for name in ['config.json', 'generation_config.json', 'tokenizer.json', 'preprocessor_config.json']:
        assert (SETTINGS / name).is_file(), f'기존 처리 설정 파일이 없습니다: {name}'
    subjects = list(read(BASE_RESULT / '_summary.json')['subject_accuracy'])
    assert len(subjects) == 30
    assert sum(row['correct'] for row in results(BASE_RESULT, subjects).values()) == 480
    assert sum(row['correct'] for row in results(A_RESULT, subjects).values()) == 501
    missing = sum(not (MERGED_ROOT / label / 'merge_record.json').is_file() for label in LABELS)
    required = (missing * 9 + 8) * 1024**3
    assert shutil.disk_usage(PROJECT).free >= required, f'디스크 여유 공간이 약 {required / 1024**3:.0f} GiB 필요합니다.'
    return subjects

def merge(label):
    sys.path.insert(0, str(DEPENDENCIES))
    import torch
    import peft
    import accelerate
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from transformers import Qwen3VLForConditionalGeneration
    assert peft.__version__ == '0.17.1' and accelerate.__version__ == '1.7.0'
    assert Path(peft.__file__).resolve().is_relative_to(DEPENDENCIES.resolve())
    assert torch.cuda.is_available(), '개인 컴퓨터 GPU를 사용할 수 없습니다.'
    assert '4080' in torch.cuda.get_device_name(0), '개인 컴퓨터의 RTX 4080 환경인지 확인해 주세요.'
    manifest = read(PACKAGE / 'manifest.json')
    adapter = PACKAGE / manifest['models'][label]['adapter_path']
    assert digest(adapter / 'adapter_model.safetensors') == manifest['files'][manifest['models'][label]['adapter_path'] + '/adapter_model.safetensors']['sha256']
    destination = MERGED_ROOT / label
    assert not destination.exists(), '병합 목적지에 기존 파일이 있습니다.'
    assert shutil.disk_usage(PROJECT).free >= 12 * 1024**3
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=label + '_building_', dir=MERGED_ROOT))
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map={'': 'cuda:0'}, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    config = model.peft_config['default']
    assert config.r == 8 and config.lora_alpha == 16
    assert config.bias == 'none' and config.modules_to_save is None
    assert not config.use_dora and not config.use_rslora
    scales = {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLayer) and 'default' in layer.scaling:
            assert 'language_model' in name and 'visual' not in name
            assert abs(layer.scaling['default'] - 2.0) < 1e-9
            scales[name] = layer.scaling['default']
    assert len(scales) == 144, f'LoRA 대상 수 불일치: {len(scales)}'
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size='4GB')
    copied = {}
    for path in SETTINGS.iterdir():
        if path.is_file() and path.suffix in {'.json', '.txt', '.jinja', '.model', '.tiktoken'} and path.name not in {'merge_record.json', 'model.safetensors.index.json'}:
            shutil.copy2(path, staging / path.name)
            assert digest(staging / path.name) == digest(path)
            copied[path.name] = digest(path)
    index = read(staging / 'model.safetensors.index.json')
    assert set(index['weight_map']) == set(read(BASE / 'model.safetensors.index.json')['weight_map'])
    assert not any('lora_' in key for key in index['weight_map'])
    weights = sorted(set(index['weight_map'].values()))
    record = {'모델': label, '원본 revision': BASE_REVISION, '어댑터 SHA256': digest(adapter / 'adapter_model.safetensors'), '병합 정밀도': 'BF16', '적용 배율': 1.0, '실제 LoRA 배율': 2.0, '대상 모듈 수': len(scales), 'PEFT 버전': peft.__version__, '설정 복사 원본': str(SETTINGS), '처리 설정 SHA256': copied, '가중치 SHA256': {name: digest(staging / name) for name in weights}}
    save(staging / 'merge_record.json', record)
    staging.rename(destination)
    print('병합 완료:', destination, flush=True)

def verify_merged(label, manifest):
    folder = MERGED_ROOT / label
    record = read(folder / 'merge_record.json')
    adapter_name = manifest['models'][label]['adapter_path'] + '/adapter_model.safetensors'
    assert record['어댑터 SHA256'] == manifest['files'][adapter_name]['sha256']
    assert record['적용 배율'] == 1.0 and record['대상 모듈 수'] == 144
    for name, expected in record['가중치 SHA256'].items():
        assert digest(folder / name) == expected, f'병합 가중치 변경: {name}'
    for name, expected in record['처리 설정 SHA256'].items():
        assert digest(folder / name) == expected

def paired(left, right):
    assert set(left) == set(right)
    recovered = regressed = 0
    for key, row in right.items():
        assert row['answer'] == left[key]['answer']
        recovered += int(not left[key]['correct'] and row['correct'])
        regressed += int(left[key]['correct'] and not row['correct'])
    discordant = recovered + regressed
    pvalue = 1.0 if discordant == 0 else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1)) / 2**discordant)
    return {'회복': recovered, '퇴보': regressed, '정답 순증가': recovered-regressed, 'McNemar exact p': pvalue}

def compare(subjects, completed):
    baseline = results(BASE_RESULT, subjects)
    original = results(A_RESULT, subjects)
    report = {'base': {'정답 수': 480, '전체': 900}, 'A': {'정답 수': 501, '전체': 900, '순정 대비': paired(baseline, original)}}
    for label in completed:
        values = results(RESULT_ROOT / label, subjects)
        report[label] = {'정답 수': sum(row['correct'] for row in values.values()), '전체': 900, '순정 대비': paired(baseline, values), 'A 대비': paired(original, values), '과목별 정답 수': {subject: sum(row['correct'] for row in values.values() if row['subject'] == subject) for subject in subjects}}
    if {'native_reasoning', 'answer_control'} <= set(completed):
        report['풀이형 대 정답형 대조군'] = paired(results(RESULT_ROOT / 'answer_control', subjects), results(RESULT_ROOT / 'native_reasoning', subjects))
    report['해석 제한'] = '고정된 사용자 평가 코드의 MMMU Validation 결과입니다. 공식 점수와 동일하다고 주장하지 않습니다. 다중 후보 탐색을 포함하므로 작은 차이로 우월성을 확정하지 않습니다.'
    save(JOB / 'mmmu_comparison.json', report)
    return report

def main():
    global ACTIVE_LOG
    import fcntl
    with (JOB / 'batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('같은 평가 묶음이 이미 실행 중입니다.')
        try:
            status('평가 환경과 전송 파일 확인 중')
            subjects = check()
            manifest = unpack()
            save(JOB / 'requirements_evaluation.txt.json', {'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), 'python': sys.version, 'ninja': shutil.which('ninja')})
            completed = []
            for label in LABELS:
                folder = MERGED_ROOT / label
                if not (folder / 'merge_record.json').is_file():
                    ACTIVE_LOG = JOB / f'{label}_merge.log'
                    status('BF16 모델 병합 중', 모델=label, 로그=str(ACTIVE_LOG))
                    with ACTIVE_LOG.open('a', encoding='utf-8') as log:
                        subprocess.run([sys.executable, '-u', str(JOB / 'run.py'), '--merge', label], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_merged(label, manifest)
                target = RESULT_ROOT / label
                target.mkdir(parents=True, exist_ok=True)
                fixed_json(target / 'evaluation_binding.json', {'모델': str(folder), '어댑터 SHA256': read(folder / 'merge_record.json')['어댑터 SHA256'], '평가 코드 SHA256': EVALUATOR_SHA, '데이터 revision': DATA_REVISION})
                for subject in subjects:
                    path = target / f'{subject}.json'
                    if path.is_file():
                        values = read(path)
                        reference = {row['id']: row for row in read(BASE_RESULT / f'{subject}.json')}
                        assert len(values) == 30 and {row['id'] for row in values} == set(reference)
                        assert all(row['answer'] == reference[row['id']]['answer'] for row in values)
                if not (target / '_summary.json').is_file():
                    assert digest(EVALUATOR) == EVALUATOR_SHA
                    ACTIVE_LOG = JOB / f'{label}_evaluation.log'
                    status('MMMU 900개 평가 중', 모델=label, 순서=f'{len(completed)+1}/3', 로그=str(ACTIVE_LOG), 결과폴더=str(target))
                    command = [sys.executable, '-u', str(EVALUATOR), '--model_path', str(folder), '--model_revision', BASE_REVISION, '--data_root', 'MMMU/MMMU', '--data_revision', DATA_REVISION, '--out_dir', str(target)]
                    save(JOB / f'{label}_evaluation_command.json', command)
                    with ACTIVE_LOG.open('a', encoding='utf-8') as log:
                        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    assert digest(EVALUATOR) == EVALUATOR_SHA
                summary = read(target / '_summary.json')
                for key, expected in {'max_new_tokens':512, 'min_pixels':200704, 'max_pixels':802816, 'seed':3407, 'presence_penalty':1.5, 'backend':'vllm', 'model_path':str(folder)}.items():
                    assert summary[key] == expected, f'평가 설정 불일치: {key}'
                assert len(summary['subject_accuracy']) == 30
                results(target, subjects)
                shutil.copy2(folder / 'merge_record.json', target / 'merge_record.json')
                completed.append(label)
                report = compare(subjects, completed)
                print(label, 'MMMU 정답:', report[label]['정답 수'], '/900', flush=True)
            status('세 모델의 MMMU 평가와 비교 완료', 결과=compare(subjects, completed), 결과파일=str(JOB / 'mmmu_comparison.json'))
            print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc(), 로그=str(ACTIVE_LOG))
            raise

def launch():
    assert PROJECT.is_dir(), '개인 컴퓨터 WSL에서 실행해 주세요.'
    JOB.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    target = JOB / 'run.py'
    if target.exists():
        assert target.read_bytes() == source, '기존 실행 파일과 내용이 다릅니다.'
    else:
        target.write_bytes(source)
    with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
        log.write('\n=== MMMU 세 모델 순차 실행 요청 ===\n')
        log.flush()
        process = subprocess.Popen([sys.executable, '-u', str(target)], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('=== 개인 컴퓨터 백그라운드 실행 요청 완료 ===')
    print('프로세스 번호:', process.pid)
    print('작업 폴더:', JOB)
    print('로그 파일:', JOB / 'runner.log')

if __name__ == '__main__':
    os.environ['PATH'] = str(PROJECT / 'venv_vllm/bin') + os.pathsep + os.environ.get('PATH', '')
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--merge', choices=LABELS)
    arguments = parser.parse_args()
    if arguments.launch:
        launch()
    elif arguments.merge:
        merge(arguments.merge)
    else:
        main()
