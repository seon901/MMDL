from pathlib import Path
from datetime import datetime
import fcntl
import gc
import json
import os
import re
import traceback
import unicodedata
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
ROOT = Path('/shared_dir/mmdl')
JOB = Path('/shared_dir/mmdl/checkpoints/replay_reasoning_compare_20261120_212449_530854')
PROFILE = ROOT / 'checkpoints/text_replay_profile_20261119_214637_363626'
BASE = ROOT / 'models/Qwen3-VL-4B-Instruct'
SUBJECTS = ['language science', 'natural science', 'social science']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save(JOB / 'text_status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())

def accepted(value, choices):
    value = value.strip()
    for wrapper in ['**', '__', '`']:
        if len(value) > 2 * len(wrapper) and value.startswith(wrapper) and value.endswith(wrapper):
            value = value[len(wrapper):-len(wrapper)].strip()
    match = re.fullmatch('(?:\\(([A-Z])\\)|([A-Z]))(?:[.):]\\s*(.*)|\\s*)', value, flags=re.I | re.S)
    label, tail = (None, value)
    if match:
        candidate = (match.group(1) or match.group(2)).upper()
        label = candidate if ord(candidate) - 65 < len(choices) else None
        tail = (match.group(3) or '').strip()
    matches = [chr(65 + index) for index, text in enumerate(choices) if norm(text) == norm(tail)]
    text_label = matches[0] if len(matches) == 1 else None
    if label is not None and (not tail or text_label == label):
        return label
    if not match and text_label is not None:
        return text_label
    return None

def main():
    rows = read(PROFILE / 'internal_dev.json')
    provenance = read(PROFILE / 'internal_dev_provenance.json')
    assert len(rows) == len(provenance) == 300
    raw_root = ROOT / 'data/mmdl_source_20261114_220144_576558'
    entry = next((item for item in read(raw_root / 'dataset_manifest.json')['sources'] if item['tag'] == 'scienceqa_train'))
    assert len(entry['files']) == 1
    raw = pq.read_table(raw_root / entry['files'][0], columns=['choices', 'answer']).to_pylist()
    predictions = {}
    for label, filename in [('base', 'base_internal_dev_predictions.json'), ('A', 'A_internal_dev_predictions.json')]:
        values = read(PROFILE / filename)
        assert len(values) == 300
        converted = []
        for position, item in enumerate(values):
            record = provenance[position]
            assert item['input_sha256'] == record['입력 SHA256']
            original = raw[int(record['원본 행 번호'])]
            gold = chr(65 + original['answer'])
            assert item['gold'] == gold
            parsed = accepted(item['predicted'], original['choices'])
            assert parsed is not None
            converted.append({**item, 'accepted': parsed, 'correct': parsed == gold})
        predictions[label] = converted
    assert sum((item['correct'] for item in predictions['base'])) == 252
    assert sum((item['correct'] for item in predictions['A'])) == 257
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    for label in ['answer_control', 'native_reasoning', 'replay_answer_control', 'replay_native_reasoning']:
        folder = Path(ADAPTERS[label])
        assert read(folder / 'trainer_state.json')['global_step'] == 500
        path = JOB / f'text_{label}_predictions.json'
        values = read(path) if path.is_file() else []
        assert len(values) <= 300
        for position, item in enumerate(values):
            assert item['input_sha256'] == provenance[position]['입력 SHA256']
            original = raw[int(provenance[position]['원본 행 번호'])]
            assert item['gold'] == chr(65 + original['answer'])
            assert item['accepted'] == accepted(item['predicted'], original['choices'])
        if len(values) < 300:
            status('텍스트 개발 모델 로딩 중', 구성=label)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
            model = PeftModel.from_pretrained(model, folder, is_trainable=False, local_files_only=True).to('cuda').eval()
            model.config.use_cache = True
            for position in range(len(values), 300):
                record = provenance[position]
                original = raw[int(record['원본 행 번호'])]
                gold = chr(65 + original['answer'])
                prompt = rows[position]['conversations'][0]['value']
                assert rows[position]['conversations'][-1]['value'] == 'Answer: ' + gold
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
                assert inputs['input_ids'].shape[-1] + 128 <= 2048
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1, repetition_penalty=1.0, use_cache=True)
                tokens = generated[0, inputs['input_ids'].shape[-1]:]
                text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                matches = re.findall('^\\s*Answer\\s*:\\s*(.*?)\\s*$', text, flags=re.I | re.M)
                predicted = matches[-1].strip() if matches else ''
                parsed = accepted(predicted, original['choices'])
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                values.append({'input_sha256': record['입력 SHA256'], 'subject': record['과목'], 'gold': gold, 'predicted': predicted, 'accepted': parsed, 'correct': parsed == gold, 'raw': text, 'exhausted': len(tokens) >= 128 and int(tokens[-1]) not in eos_ids})
                del inputs, generated, tokens
                if (position + 1) % 10 == 0 or position == 299:
                    save(path, values)
                    status('텍스트 개발 평가 중', 구성=label, 완료=position + 1, 전체=300)
                    print(f'{label}: {position + 1}/300', flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        predictions[label] = values
    results = []
    for label, values in predictions.items():
        base = predictions['base']
        trained = predictions['A']
        results.append({'구성': label, '전체': 300, '정답 수': sum((item['correct'] for item in values)), '순정 오답에서 회복': sum((not before['correct'] and item['correct'] for before, item in zip(base, values))), '순정 정답에서 퇴보': sum((before['correct'] and (not item['correct']) for before, item in zip(base, values))), 'A 오답에서 회복': sum((not before['correct'] and item['correct'] for before, item in zip(trained, values))), 'A 정답에서 퇴보': sum((before['correct'] and (not item['correct']) for before, item in zip(trained, values))), '내용 판정 보류': sum((item['accepted'] is None for item in values)), '답 형식 누락': sum((not item['predicted'] for item in values)), '길이 한도 도달': sum((item['exhausted'] for item in values)), '과목별 정답 수': {tag: sum((item['correct'] for item in values if item['subject'] == tag)) for tag in SUBJECTS}})
    image_results = read(JOB / 'image_comparison.json')['결과']
    save(JOB / 'combo_comparison.json', {'이미지 개발 400개': image_results, '텍스트 개발 300개': results, '학습 구성': read(JOB / 'combo_plan.json'), '해석 제한': '이미지 문항 교체와 Replay 추가를 함께 적용했습니다. 텍스트 채점은 답 줄의 번호와 선택지 내용을 확인합니다. MMMU 평가 코드는 변경하지 않았습니다. 전체 MMMU 퇴보 억제는 별도 평가가 필요합니다.'})
    compact = []
    for item in image_results:
        text_result = next((result for result in results if result['구성'] == item['구성']))
        compact.append({'구성': item['구성'], '이미지 정답 수': item['정답 수'], '이미지 순정 정답 퇴보': item['순정 정답에서 퇴보'], '텍스트 정답 수': text_result['정답 수'], '텍스트 순정 정답 퇴보': text_result['순정 정답에서 퇴보'], '텍스트 A 정답 퇴보': text_result['A 정답에서 퇴보'], '텍스트 내용 판정 보류': text_result['내용 판정 보류']})
    status('조합 이미지와 텍스트 평가 완료', 결과=compact, 결과파일=str(JOB / 'combo_comparison.json'))
    print(json.dumps(read(JOB / 'text_status.json'), ensure_ascii=False, indent=2), flush=True)
ADAPTERS = {'answer_control': '/shared_dir/mmdl/checkpoints/native_reasoning_compare_20261119_232833_892795/answer_control', 'native_reasoning': '/shared_dir/mmdl/checkpoints/native_reasoning_compare_20261119_232833_892795/native_reasoning', 'replay_answer_control': '/shared_dir/mmdl/checkpoints/replay_reasoning_compare_20261120_212449_530854/replay_answer_control', 'replay_native_reasoning': '/shared_dir/mmdl/checkpoints/replay_reasoning_compare_20261120_212449_530854/replay_native_reasoning'}
if __name__ == '__main__':
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 비교 작업이 실행 중입니다. 중복 실행을 중단합니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
