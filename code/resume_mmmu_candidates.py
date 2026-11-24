from pathlib import Path
from datetime import datetime
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
JOB = PROJECT / 'logs/mmmu_candidates_20261120_004215_155629'
CUDA = Path('/usr/local/cuda-12.8')
EVALUATOR = PROJECT / 'scripts/eval_mmmu_vllm.py'
EXPECTED = '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'

def configure():
    compiler = CUDA / 'bin/nvcc'
    if not compiler.is_file():
        print('발견한 CUDA 경로:', flush=True)
        for path in sorted(Path('/usr/local').glob('cuda*')):
            print(path, flush=True)
        raise RuntimeError('기존 CUDA 12.8 컴파일러를 찾지 못했습니다. 이 출력을 보내주세요.')
    version = subprocess.check_output([str(compiler), '--version'], text=True)
    assert re.search(r'release 12\.8[,\s]', version), 'CUDA 경로와 실제 컴파일러 버전이 다릅니다.'
    assert (CUDA / 'lib64').is_dir()
    os.environ['CUDA_HOME'] = str(CUDA)
    os.environ['CUDA_PATH'] = str(CUDA)
    os.environ['PATH'] = str(PROJECT / 'venv_vllm/bin') + os.pathsep + str(CUDA / 'bin') + os.pathsep + os.environ.get('PATH', '')
    old_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = str(CUDA / 'lib64') + (os.pathsep + old_library_path if old_library_path else '')
    os.environ['MAX_JOBS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    assert os.environ.get('VLLM_USE_FLASHINFER_SAMPLER') in (None, '1'), '샘플러 환경변수가 기존 FlashInfer 경로와 다릅니다. 현재 값을 확인해야 합니다.'
    print('CUDA_HOME:', os.environ['CUDA_HOME'], flush=True)
    print(version.strip(), flush=True)
    return version

def sampler_test():
    configure()
    print('=== GPU 샘플러 사전 검사 모듈 로딩 ===', flush=True)
    import torch
    from flashinfer.jit.cpp_ext import get_cuda_path
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    assert torch.cuda.is_available()
    assert torch.version.cuda == '12.8', f'예상한 PyTorch CUDA 버전과 다릅니다: {torch.version.cuda}'
    assert Path(get_cuda_path()).resolve() == CUDA.resolve()
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('FlashInfer 컴파일러 경로:', get_cuda_path(), flush=True)
    sampler = TopKTopPSampler()
    assert sampler.forward.__name__ == 'forward_cuda', '기존 FlashInfer 샘플러 경로를 선택하지 못했습니다.'
    torch.manual_seed(3407)
    logits = torch.randn(4, 256, device='cuda', dtype=torch.float32)
    k = torch.full((4,), 20, device='cuda', dtype=torch.int32)
    p = torch.full((4,), 0.8, device='cuda', dtype=torch.float32)
    print('=== FlashInfer 커널 컴파일과 실행 검사 ===', flush=True)
    print('처음에는 컴파일 시간이 필요합니다.', flush=True)
    sampled, _ = sampler(logits.clone(), {}, k, p)
    torch.cuda.synchronize()
    assert tuple(sampled.shape) == (4,)
    assert bool(((sampled >= 0) & (sampled < 256)).all())
    generators = {i: torch.Generator(device='cuda').manual_seed(3407) for i in range(4)}
    sampled, _ = sampler(logits.clone(), generators, k, p)
    torch.cuda.synchronize()
    assert tuple(sampled.shape) == (4,)
    assert bool(((sampled >= 0) & (sampled < 256)).all())
    print('=== GPU 샘플러 사전 검사 통과 ===', flush=True)

def main():
    assert PROJECT.is_dir() and JOB.is_dir(), '개인 컴퓨터 WSL의 작업 폴더가 필요합니다.'
    assert hashlib.sha256(EVALUATOR.read_bytes()).hexdigest() == EXPECTED, '기존 평가 코드가 변경됐습니다.'
    with (JOB / 'batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('기존 배치가 아직 실행 중입니다. 중복 실행하지 않습니다.')
        version = configure()
        log_path = JOB / ('sampler_preflight_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--sampler-test']
        print('=== 사전 검사 시작 ===', flush=True)
        print('로그:', log_path, flush=True)
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            with log_path.open('r', encoding='utf-8', errors='replace') as reader:
                started = time.monotonic()
                last_notice = started
                while True:
                    output = reader.read()
                    if output:
                        print(output, end='', flush=True)
                    code = process.poll()
                    if code is not None:
                        remaining = reader.read()
                        if remaining:
                            print(remaining, end='', flush=True)
                        assert code == 0, f'사전 검사가 실패했습니다. 배치를 시작하지 않았습니다. 로그: {log_path}'
                        break
                    now = time.monotonic()
                    if now - last_notice >= 30:
                        print(f'사전 검사 진행 중: {int(now-started)}초 경과', flush=True)
                        last_notice = now
                    time.sleep(2)
        record = {
            '수정 시각': datetime.now().isoformat(),
            '원인': 'CUDA 12.1 nvcc가 FlashInfer의 compress-mode 옵션을 지원하지 않음',
            '복구 CUDA_HOME': str(CUDA),
            '컴파일러 버전': version,
            '사전 검사 로그': str(log_path),
            '사전 검사': '통과',
            '평가 코드 SHA256': EXPECTED,
            '평가 코드 변경': False,
            '샘플러 변경': False,
            '설치 패키지 변경': False,
            '패키지 버전': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'vllm', 'flashinfer-python']},
        }
        (JOB / 'cuda_recovery.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== 기존 순차 평가 재개 요청 ===', flush=True)
    subprocess.run([sys.executable, str(PROJECT / 'scripts/run_mmmu_candidates.py'), '--launch'], check=True)
    print('병합이 완료된 첫 모델은 기존 병합 결과를 사용합니다.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler-test', action='store_true')
    arguments = parser.parse_args()
    if arguments.sampler_test:
        sampler_test()
    else:
        main()