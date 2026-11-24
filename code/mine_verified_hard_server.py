from pathlib import Path
from datetime import datetime
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
import re
import traceback
import unicodedata

ROOT = Path('/shared_dir/mmdl')
DATA = ROOT / 'data/engineering83_prepared_20261118/paired_candidate_split_20261119_085350_215640'
VALIDATION = ROOT / 'data/engineering83_prepared_20261118/validation_20261118_230546_267103'
GROUPS = VALIDATION / 'contamination_20261118_231821_678269/split_groups.json'
ADAPTER = ROOT / 'checkpoints/replay_compare_20261119_214637_363626/R20'
BASE = ROOT / 'models/Qwen3-VL-4B-Instruct'
JOB = ROOT / 'checkpoints/verified_hard_profile_20261121'
SUFFIX = 'Solve the problem and end with a line beginning with Answer:.'
DIRECT = 'Give only your final answer on one line beginning with Answer:.'

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split()).strip()

def main():
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    train = read(DATA / 'answer_only_train.json')
    dev = read(DATA / 'answer_only_internal_dev.json')
    assert len(train) == 63 and len(dev) == 16
    train_ids = {row['id'] for row in train}
    dev_ids = {row['id'] for row in dev}
    assert len(train_ids) == 63 and len(dev_ids) == 16 and not train_ids & dev_ids
    mapping = {}
    for group in read(GROUPS):
        for candidate in group['members']:
            assert candidate not in mapping
            mapping[candidate] = group['group_id']
    assert train_ids | dev_ids <= set(mapping)
    assert not {mapping[x] for x in train_ids} & {mapping[x] for x in dev_ids}
    assert sha(ADAPTER / 'adapter_model.safetensors') == '4f23f6ecd950630fe9d6208b42550e166242a790d37aaa6fe50bb5975df49784'
    plan = {'목적': '검토된 공학 학습 문항의 R20 난도 진단',
            '학습 후보 수': 63, '개발 문항 선별 제외': 16, '모델': str(ADAPTER),
            '학습 JSON SHA256': sha(DATA / 'answer_only_train.json'),
            '어댑터 SHA256': sha(ADAPTER / 'adapter_model.safetensors'),
            '정답형 지시문': DIRECT, '출력 한도': 256, 'min_pixels': 3136, 'max_pixels': 200704,
            '선별 원칙': '정답 손실이 높은 학습 문항을 먼저 확인. 단순 표기 불일치를 확정 오답으로 해석하지 않음.',
            'teacher 및 유료 API': False, '현재 단계': '선별만 수행. 학습하지 않음.'}
    plan_file = JOB / 'profile_plan.json'
    if plan_file.exists():
        assert read(plan_file) == plan, '이전 선별 실행과 조건이 다릅니다.'
    else:
        save(plan_file, plan)
    saved = JOB / 'predictions.json'
    values = read(saved) if saved.exists() else []
    assert len(values) <= 63
    processor = AutoProcessor.from_pretrained(BASE, min_pixels=3136, max_pixels=200704, local_files_only=True)
    model = None
    for index, row in enumerate(train):
        image_path = Path(row['image'])
        if not image_path.is_absolute():
            image_path = DATA / image_path
        assert image_path.is_file(), f'이미지 파일이 없습니다: {image_path}'
        assert row['conversations'][0]['from'] == 'human' and row['conversations'][-1]['from'] == 'gpt'
        original_prompt = row['conversations'][0]['value']
        assert original_prompt.count('<image>') == 1
        assert original_prompt.endswith(SUFFIX), '기존 지시문 구조가 다릅니다.'
        prompt = original_prompt[:-len(SUFFIX)].replace('<image>', '').strip() + '\n' + DIRECT
        target = row['conversations'][-1]['value'].strip()
        assert target.startswith('Answer: ') and len(target) > len('Answer: ')
        binding = hashlib.sha256(json.dumps([row['id'], prompt, target, sha(image_path)], ensure_ascii=False).encode('utf-8')).hexdigest()
        if index < len(values):
            assert values[index]['id'] == row['id'] and values[index]['input_sha256'] == binding
            continue
        if model is None:
            status('R20 모델 로딩 중')
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
            model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False, local_files_only=True).to('cuda').eval()
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': target}], tokenize=False, add_generation_prompt=False)
        prefix = processor(text=[prompt_text], images=[image], return_tensors='pt')
        full = processor(text=[full_text], images=[image], return_tensors='pt')
        image.close()
        length = prefix['input_ids'].shape[1]
        assert length + 256 <= 2048 and full['input_ids'].shape[1] <= 2048
        assert torch.equal(full['input_ids'][0, :length], prefix['input_ids'][0]), '정답 토큰 경계를 확인하지 못했습니다.'
        labels = full['input_ids'].clone()
        labels[:, :length] = -100
        full = full.to('cuda')
        with torch.inference_mode():
            loss = float(model(**full, labels=labels.to('cuda'), use_cache=False).loss)
        assert math.isfinite(loss)
        del full, labels
        prefix = prefix.to('cuda')
        with torch.inference_mode():
            generated = model.generate(**prefix, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1, repetition_penalty=1.0, use_cache=True)
        tokens = generated[0, length:]
        raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        answers = re.findall(r'^\s*Answer\s*:\s*(.*?)\s*$', raw, flags=re.I | re.M)
        extracted = answers[-1].strip() if answers else ''
        gold = target[len('Answer: '):].strip()
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        exhausted = len(tokens) >= 256 and int(tokens[-1]) not in eos_ids
        values.append({'id': row['id'], 'group': mapping[row['id']], 'input_sha256': binding,
                       'question': prompt, 'gold': gold, 'prediction': extracted, 'raw': raw,
                       'gold_loss': loss, 'exact_normalized_match': bool(extracted) and norm(extracted) == norm(gold),
                       'missing_answer': not extracted, 'length_limit': exhausted})
        save(saved, values)
        status('학습 문항 난도 진단 중', 완료=len(values), 전체=63, 로그=str(JOB / 'runner.log'))
        print(f'{len(values)}/63 완료', flush=True)
        del prefix, generated, tokens
    ordered = sorted(values, key=lambda item: (-item['gold_loss'], item['id']))
    by_id = {row['id']: row for row in train}
    selected = ordered[:min(32, len(ordered))]
    save(JOB / 'ranked_candidates.json', ordered)
    save(JOB / 'provisional_answer_only.json', [by_id[item['id']] for item in selected])
    save(JOB / 'provisional_reasoning_ids.json', [item['id'] for item in selected])
    report = {'학습 후보 검사 수': len(values), '정답 표기 일치': sum(r['exact_normalized_match'] for r in values),
              '답 형식 누락': sum(r['missing_answer'] for r in values), '길이 한도 도달': sum(r['length_limit'] for r in values),
              '정답 손실 상위 잠정 후보': len(selected), '개발 문항 사용': 0,
              '주의': '표기 불일치는 확정 오답 수가 아닙니다. 수식 동치와 답 형식 문제를 구분한 후 보강 대상을 정합니다.',
              '학습 실행': False, '결과 폴더': str(JOB)}
    save(JOB / 'profile_summary.json', report)
    status('학습 문항 난도 진단 완료', 결과=report)
    print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 학교 GPU 비교 작업이 실행 중입니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
