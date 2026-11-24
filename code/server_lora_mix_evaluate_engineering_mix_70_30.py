import sys
sys.path.insert(0, '/shared_dir/mmdl')
from mine_verified_hard_server import *
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
JOB = Path('/shared_dir/mmdl/checkpoints/lora_mix_20261121')
def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})
ADAPTER = Path('/shared_dir/mmdl/checkpoints/lora_mix_20261121/adapters/mix_70_30')
train = read(DATA / 'answer_only_internal_dev.json')
assert len(train) == 16
mapping = {candidate: group['group_id'] for group in read(GROUPS) for candidate in group['members']}
saved = Path('/shared_dir/mmdl/checkpoints/lora_mix_20261121/engineering_mix_70_30.json')
values = read(saved) if saved.exists() else []
assert len(values) <= 16
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
    answers = re.findall('^\\s*Answer\\s*:\\s*(.*?)\\s*$', raw, flags=re.I | re.M)
    extracted = answers[-1].strip() if answers else ''
    gold = target[len('Answer: '):].strip()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    exhausted = len(tokens) >= 256 and int(tokens[-1]) not in eos_ids
    values.append({'id': row['id'], 'group': mapping[row['id']], 'input_sha256': binding, 'question': prompt, 'gold': gold, 'prediction': extracted, 'raw': raw, 'gold_loss': loss, 'exact_normalized_match': bool(extracted) and norm(extracted) == norm(gold), 'missing_answer': not extracted, 'length_limit': exhausted})
    save(saved, values)
    status('학습 문항 난도 진단 중', 완료=len(values), 전체=16, 로그=str(JOB / 'runner.log'))
    print(f'{len(values)}/16 완료', flush=True)
    del prefix, generated, tokens
