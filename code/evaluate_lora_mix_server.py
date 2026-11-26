from pathlib import Path
from datetime import datetime
import copy
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import traceback

ROOT = Path('/shared_dir/mmdl')
OLD = ROOT / 'checkpoints/replay_compare_20261119_214637_363626'
PREVIOUS = ROOT / 'checkpoints/verified_hard_compare_20261121'
SOURCES = {
    'R20': OLD / 'R20',
    'reasoning': ROOT / 'checkpoints/verified_hard_reasoning_20261121/hard_reasoning',
}
EXPECTED = {
    'R20': '4f23f6ecd950630fe9d6208b42550e166242a790d37aaa6fe50bb5975df49784',
    'reasoning': '33a3738a702a69dd150e7ee89acc7694e7f7bf35ecf5d1637f27c4f4b7d2a669',
}
WEIGHTS = {'mix_90_10': (0.9,0.1), 'mix_80_20': (0.8,0.2), 'mix_70_30': (0.7,0.3)}
JOB = ROOT / 'checkpoints/lora_mix_20261121'
REPO = ROOT / 'code/Qwen3-VL/qwen-vl-finetune'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def status(state,**extra):
    save(JOB/'status.json',{'상태':state,'갱신 시각':datetime.now().isoformat(),**extra})

def make_mixtures():
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(4)
    configs={name:read(folder/'adapter_config.json') for name,folder in SOURCES.items()}
    tensors={}
    for name,folder in SOURCES.items():
        assert sha(folder/'adapter_model.safetensors')==EXPECTED[name], f'원본 어댑터 변경: {name}'
        config=configs[name]
        assert config['r']==8 and config['lora_alpha']==16
        assert config['bias']=='none' and config['modules_to_save'] is None
        assert not config.get('use_rslora') and not config.get('use_dora') and not config.get('lora_bias')
        assert not config.get('rank_pattern') and not config.get('alpha_pattern')
        assert config['peft_type']=='LORA' and not config.get('fan_in_fan_out')
        tensors[name]=load_file(str(folder/'adapter_model.safetensors'),device='cpu')
    assert set(tensors['R20'])==set(tensors['reasoning'])
    assert set(configs['R20']['target_modules'])==set(configs['reasoning']['target_modules'])
    assert configs['R20']['base_model_name_or_path']==configs['reasoning']['base_model_name_or_path']
    keys=sorted(key for key in tensors['R20'] if key.endswith('.lora_A.weight'))
    assert len(keys)==144 and len(tensors['R20'])==288
    assert all('language_model' in key and 'visual' not in key for key in keys)
    parent=JOB/'adapters'
    parent.mkdir(exist_ok=True)
    adapters={}
    for label,(left,right) in WEIGHTS.items():
        destination=parent/label
        identity={'sources':{k:str(v) for k,v in SOURCES.items()},'source_sha256':EXPECTED,
                  'weights':[left,right],'r':16,'lora_alpha':16,
                  'formula':'delta_mix = weight_R20 * delta_R20 + weight_reasoning * delta_reasoning',
                  'method':'Exact low-rank concatenation. Source alpha/r is included in A; mixed alpha/r equals one. No factor averaging, no SVD truncation and no training.'}
        if destination.exists():
            record=read(destination/'mix_record.json')
            assert record['identity']==identity
            assert record['adapter_sha256']==sha(destination/'adapter_model.safetensors')
            assert record['config_sha256']==sha(destination/'adapter_config.json')
            adapters[label]=destination
            continue
        mixed={}
        maximum_error=0.0
        generator=torch.Generator(device='cpu').manual_seed(42)
        for a_key in keys:
            b_key=a_key.replace('.lora_A.weight','.lora_B.weight')
            a0=tensors['R20'][a_key].float()
            b0=tensors['R20'][b_key].float()
            a1=tensors['reasoning'][a_key].float()
            b1=tensors['reasoning'][b_key].float()
            assert a0.shape==a1.shape and b0.shape==b1.shape
            assert a0.shape[0]==b0.shape[1]==8
            a=torch.cat([a0*(left*2.0),a1*(right*2.0)],dim=0).contiguous()
            b=torch.cat([b0,b1],dim=1).contiguous()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            probe=torch.randn(a0.shape[1],2,generator=generator)
            expected=left*2.0*(b0@(a0@probe))+right*2.0*(b1@(a1@probe))
            actual=b@(a@probe)
            assert torch.allclose(actual,expected,rtol=1e-4,atol=1e-5), f'가중합 검사 실패: {a_key}'
            maximum_error=max(maximum_error,float((actual-expected).abs().max()))
            mixed[a_key]=a
            mixed[b_key]=b
        stage=Path(tempfile.mkdtemp(prefix=label+'_building_',dir=parent))
        config=copy.deepcopy(configs['R20'])
        config.update(r=16,lora_alpha=16,inference_mode=True)
        save(stage/'adapter_config.json',config)
        save_file(mixed,str(stage/'adapter_model.safetensors'),metadata={'format':'pt'})
        save(stage/'mix_record.json',{'identity':identity,'verified_modules':144,'maximum_probe_error':maximum_error,
             'adapter_sha256':sha(stage/'adapter_model.safetensors'),'config_sha256':sha(stage/'adapter_config.json')})
        stage.rename(destination)
        adapters[label]=destination
        print(label,'어댑터 혼합과 144개 모듈 검사 완료',flush=True)
    return adapters

def main():
    path=ROOT/'train_verified_hard_server.py'
    spec=importlib.util.spec_from_file_location('mix_evaluation_helper',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.JOB=JOB
    helper=module.load_helper()
    save(JOB/'experiment_plan.json',{'목적':'R20 중심의 LoRA 변화분 가중합 비교','weights':WEIGHTS,
         '기준':'기존 R20','추가 학습':False,'시드':42,'혼합 rank':16,'혼합 alpha':16,
         '설명':'두 rank 8 변화분의 가중합을 rank 16으로 정확하게 표현. 별도의 rank 16 학습 실험이 아님.',
         '선택':'이미지 및 텍스트 개발 성능과 퇴보를 함께 비교. MMMU 과목별 정답을 혼합 규칙에 사용하지 않음.'})
    status('LoRA 변화분 혼합 중')
    adapters={'reference_R20':SOURCES['R20'],**make_mixtures()}
    images={'reference_R20':read(OLD/'evaluation_R20/R20_predictions.json')}
    for label,adapter in adapters.items():
        if label=='reference_R20':continue
        output=JOB/('image_'+label)
        script=JOB/('evaluate_'+label+'.py')
        helper.snapshot(script,helper.image_code(label,adapter,output))
        predictions=output/(label+'_predictions.json')
        if not (predictions.is_file() and len(read(predictions))==400 and (output/'comparison.json').is_file()):
            status('혼합 모델 이미지 개발 평가 중',구성=label,로그=str(JOB/(label+'_evaluation.log')))
            module.execute(script,JOB/(label+'_evaluation.log'))
        images[label]=read(predictions)
    module.fixed_json(JOB/'text_reference_R20_predictions.json',read(OLD/'text_R20_predictions.json'))
    script=JOB/'evaluate_text.py'
    helper.snapshot(script,helper.text_code(adapters))
    status('혼합 모델 텍스트 개발 평가 중',로그=str(JOB/'text_evaluation.log'))
    module.execute(script,JOB/'text_evaluation.log')
    texts=read(JOB/'text_all_predictions.json')
    module.fixed_json(JOB/'engineering_reference_R20.json',read(PREVIOUS/'engineering_reference_R20.json'))
    engineering={}
    for label,adapter in adapters.items():
        script=JOB/('evaluate_engineering_'+label+'.py')
        helper.snapshot(script,module.engineering_code(label,adapter))
        status('혼합 모델 공학 개발 평가 중',구성=label,로그=str(JOB/('engineering_'+label+'.log')))
        module.execute(script,JOB/('engineering_'+label+'.log'))
        engineering[label]=read(JOB/('engineering_'+label+'.json'))
    base=read(ROOT/'checkpoints/expanded_lr_20261116_223345_306888/generation_eval_20261116_231128_946555/base_predictions.json')
    report=[]
    for label in adapters:
        iv,tv,ev=images[label],texts[label],engineering[label]
        assert len(iv)==400 and len(tv)==300 and len(ev)==16
        assert all(all(row[k]==b[k] for k in ['번호','출처','원본 행 번호','QA 번호','기준 정답']) for row,b in zip(iv,base))
        assert all(row['input_sha256']==b['input_sha256'] for row,b in zip(tv,texts['base']))
        assert all(row['input_sha256']==b['input_sha256'] for row,b in zip(ev,engineering['reference_R20']))
        report.append({'구성':label,'이미지 정답 수':sum(bool(r['정답 여부']) for r in iv),
             '이미지 순정 정답 퇴보':sum(bool(b['정답 여부']) and not r['정답 여부'] for b,r in zip(base,iv)),
             '텍스트 정답 수':sum(bool(r['correct']) for r in tv),
             '텍스트 순정 정답 퇴보':sum(bool(b['correct']) and not r['correct'] for b,r in zip(texts['base'],tv)),
             '공학 개발 평균 정답 손실':sum(r['gold_loss'] for r in ev)/16,
             '공학 개발 정답 표기 일치':sum(r['exact_normalized_match'] for r in ev),
             '공학 개발 길이 한도 도달':sum(r['length_limit'] for r in ev)})
    assert report[0]['이미지 정답 수']==259 and report[0]['텍스트 정답 수']==259
    save(JOB/'mix_comparison.json',{'결과':report,'주의':'내부 개발 탐색 결과입니다. 답변 투표와 다른 단일 모델 혼합 실험입니다. MMMU 평가는 아직 실행하지 않았습니다.'})
    status('세 혼합 비율의 개발 평가 완료',결과=report,결과파일=str(JOB/'mix_comparison.json'))
    print(json.dumps(read(JOB/'status.json'),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    JOB.mkdir(parents=True,exist_ok=True)
    with (ROOT/'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 학교 GPU 비교 작업이 실행 중입니다.')
        try:
            main()
        except Exception:
            status('오류로 중단',오류=traceback.format_exc())
            raise
