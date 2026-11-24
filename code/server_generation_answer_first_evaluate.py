def answer_first_prompt(question):
    question = question.replace('Solve the problem and end with a line beginning with Answer:.', '').strip()
    return question + '\nStart with the final answer on the first line in the format Answer: <final answer>. Then give at most two sentences of explanation. Do not repeat the question or your explanation.'
from pathlib import Path
from datetime import datetime
import fcntl
import gc
import hashlib
import json
import os
import re
import traceback
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '0'})
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
ROOT = Path('/shared_dir/mmdl')
OUT = Path(__file__).resolve().parent
RUN = OUT.parent
BASE = ROOT / 'models/Qwen3-VL-4B-Instruct'
DATA = ROOT / 'data/engineering83_prepared_20261118/paired_candidate_split_20261119_085350_215640'
METADATA = ROOT / 'data/engineering83_prepared_20261118/paired_candidates.json'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(**value):
    save('status.json', {'갱신 시각': datetime.now().isoformat(), **value})

def main():
    assert torch.cuda.is_available(), 'GPU를 사용할 수 없습니다.'
    rows = read(DATA / 'answer_only_internal_dev.json')
    metadata = {item['id']: item for item in read(METADATA)}
    assert len(rows) == 16
    assert len({row['id'] for row in rows}) == 16
    models = {'base': None, 'recipe_a': ROOT / 'checkpoints/expanded_lr_20261116_223345_306888/lr_2e-4', 'answer_only': RUN / 'answer_only', 'concise_reasoning': RUN / 'concise_reasoning'}
    for name, adapter in models.items():
        if adapter is not None:
            assert (adapter / 'adapter_config.json').is_file(), f'LoRA 설정이 없습니다: {name}'
            assert (adapter / 'adapter_model.safetensors').is_file(), f'LoRA 가중치가 없습니다: {name}'
    processor = AutoProcessor.from_pretrained(BASE, min_pixels=3136, max_pixels=200704, local_files_only=True)
    save('settings.json', {'base_model': str(BASE), 'adapters': {name: str(path) if path is not None else None for name, path in models.items()}, 'min_pixels': 3136, 'max_pixels': 200704, 'max_new_tokens': 512, 'do_sample': False, 'seed': 42, 'data_sha256': hashlib.sha256((DATA / 'answer_only_internal_dev.json').read_bytes()).hexdigest(), 'grading': 'Generation collection only; no automatic accuracy claim', 'prompt_variant': 'answer_first_then_at_most_two_sentences', 'parent_script_sha256': '094fa0c751b0e4181d4541a4288c6caa5b785c72bee99adb4d615043ba96649d', 'parent_script': '/shared_dir/mmdl/checkpoints/engineering_format_pilot_20261119_085822_757298/generation_eval_20261119_090625_E0dZcD/evaluate.py'})
    all_predictions = {}
    input_checks = {}
    summaries = {}
    for name, adapter in models.items():
        status(상태='모델 로딩 중', 모델=name, 완료=0, 전체=16)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', local_files_only=True).to('cuda')
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
        model.eval()
        predictions = []
        for index, row in enumerate(rows):
            candidate = row['id']
            image_path = Path(row['image'])
            assert image_path.is_absolute() and image_path.is_file()
            question = row['conversations'][0]['value'].replace('<image>', '').strip()
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': answer_first_prompt(question)}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            with Image.open(image_path) as image:
                inputs = processor(text=[prompt], images=[image.convert('RGB')], padding=True, return_tensors='pt')
            image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
            assert bool((inputs['input_ids'] == image_token).any()), '이미지 토큰이 없습니다.'
            fingerprint = hashlib.sha256(inputs['input_ids'].numpy().tobytes() + inputs['image_grid_thw'].numpy().tobytes() + image_path.read_bytes()).hexdigest()
            if name == 'base':
                input_checks[candidate] = fingerprint
            else:
                assert input_checks[candidate] == fingerprint, f'모델별 입력이 다릅니다: {candidate}'
            prompt_length = inputs['input_ids'].shape[1]
            inputs = inputs.to('cuda')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True, repetition_penalty=1.0, pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id)
            continuation = generated[0, prompt_length:]
            token_count = int(continuation.numel())
            text = processor.decode(continuation, skip_special_tokens=True).strip()
            answers = re.findall('(?im)^\\s*(?:\\*\\*)?Answer\\s*:\\s*(.+)$', text)
            predictions.append({'id': candidate, 'model': name, 'source': metadata[candidate]['source'], 'question': question, 'reference_answer': metadata[candidate]['final_answer'], 'reference_reasoning': metadata[candidate]['explanation'], 'generated_text': text, 'extracted_answer': answers[-1].strip() if answers else None, 'generated_tokens': token_count, 'token_limit_reached': token_count >= 512, 'input_sha256': fingerprint})
            save(f'{name}_predictions.json', predictions)
            status(상태='답변 생성 중', 모델=name, 완료=index + 1, 전체=16)
            print(f'{name}: {index + 1}/16개 완료', flush=True)
            del inputs, generated, continuation
        all_predictions[name] = predictions
        summaries[name] = {'문항 수': len(predictions), '답 형식 누락': sum((item['extracted_answer'] is None for item in predictions)), '길이 한도 도달': sum((item['token_limit_reached'] for item in predictions))}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparison = []
    for index, row in enumerate(rows):
        candidate = row['id']
        comparison.append({'id': candidate, 'source': metadata[candidate]['source'], 'question': all_predictions['base'][index]['question'], 'reference_answer': metadata[candidate]['final_answer'], 'reference_reasoning': metadata[candidate]['explanation'], 'models': {name: {key: predictions[index][key] for key in ('generated_text', 'extracted_answer', 'token_limit_reached')} for name, predictions in all_predictions.items()}})
    save('comparison.json', comparison)
    save('summary.json', summaries)
    status(상태='네 모델의 생성 완료', 결과=summaries, 결과파일=str(OUT / 'comparison.json'))
if __name__ == '__main__':
    with (RUN / 'generation_eval.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 생성 평가가 이미 실행 중입니다.')
        (RUN / 'latest_generation_eval.txt').write_text(str(OUT), encoding='utf-8')
        try:
            main()
        except Exception:
            status(상태='오류로 중단', 오류=traceback.format_exc())
            raise
