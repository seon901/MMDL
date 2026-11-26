from pathlib import Path
import hashlib
import runpy
import sys

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
EVALUATOR = PROJECT / 'scripts/eval_mmmu_vllm.py'
EXPECTED = '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'


def main():
    assert hashlib.sha256(EVALUATOR.read_bytes()).hexdigest() == EXPECTED
    sys.path.insert(0, str(PROJECT / 'scripts'))
    from resume_mmmu_candidates import configure
    configure()
    import vllm
    original = vllm.LLM

    def image_only(*args, **kwargs):
        assert kwargs.get('limit_mm_per_prompt') == {'image': 8}, '기존 이미지 제한 설정이 다릅니다.'
        assert kwargs.get('max_model_len') == 4096
        assert kwargs.get('max_num_seqs') == 4
        kwargs['limit_mm_per_prompt'] = {'image': 8, 'video': 0}
        print('이미지 전용 초기화: image=8, video=0. 기존 프롬프트와 생성 및 채점 코드를 실행합니다.', flush=True)
        return original(*args, **kwargs)

    vllm.LLM = image_only
    try:
        sys.argv[0] = str(EVALUATOR)
        runpy.run_path(str(EVALUATOR), run_name='__main__')
    finally:
        vllm.LLM = original


if __name__ == '__main__':
    main()
