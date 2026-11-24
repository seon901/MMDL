from pathlib import Path
from datetime import datetime
from collections import Counter,defaultdict
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback

PROJECT=Path('/mnt/c/Users/K/Desktop/MMDL')
JOB=PROJECT/'logs/soft_vote_pilot_20261121'
DATA=JOB/'data'
BASE_RESULTS=PROJECT/'results_vllm_v2'
MODELS={
 'R20':PROJECT/'merged_models/mmmu_candidates_20261120_004215_155629/R20',
 'hard_reasoning':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_reasoning',
 'A':PROJECT/'merged_models/recipe_a_lr_2e-4_step500_20261116_232956_544892',
 'base':Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'),
 'hard_answer':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_answer',
}
GROUPS={'three':['R20','A','base'],'five':['R20','A','base','hard_answer','hard_reasoning']}
TEMPERATURES=[0.5,1.0,2.0]
EVAL_SHA='0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
sys.path.insert(0,str(PROJECT/'scripts'))

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def fixed(p,value):
    if p.exists():assert read(p)==value,f'고정 기록과 다릅니다: {p}'
    else:save(p,value)
def status(state,**extra):save(JOB/'status.json',{'상태':state,'갱신 시각':datetime.now().isoformat(),**extra})
def helper_module():
    spec=importlib.util.spec_from_file_location('input_helper',PROJECT/'scripts/run_independent_900.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare():
    from datasets import load_dataset
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    assert sha(PROJECT/'scripts/eval_mmmu_vllm.py')==EVAL_SHA
    subjects=sorted(read(BASE_RESULTS/'_summary.json')['subject_accuracy']);assert len(subjects)==30
    plan={'purpose':'Development-only small MCQ probability-voting pilot',
          'source':'MMMU/MMMU validation','revision':'98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68',
          'sample_size':300,'per_subject':10,'calibration':150,'comparison':150,
          'selection':'Sort MCQ IDs within each subject by SHA256 of soft-vote-dev-v1 plus ID. First ten, alternating calibration and comparison.',
          'models':{name:str(path) for name,path in MODELS.items()},'temperatures':TEMPERATURES,
          'engine':{'max_model_len':4096,'gpu_memory_utilization':0.85,'max_num_seqs':4,'max_logprobs':64,'logprobs_mode':'processed_logprobs','max_new_tokens':1},
          'probability':'One-token option-letter probabilities after an Answer: prefix. Include plain and space-prefixed single-token letter variants. Constrain to valid variants and use processed_logprobs.',
          'comparison':'Compare hard voting and probability means under this same direct scoring protocol. Old generative voting is a separate reference.',
          'followup_signal':'Selected soft method, chosen by calibration NLL, must gain at least five correct comparison items over hard-five and not lose to the calibration-selected individual. It must also beat the old generative-five reference by at least three comparison items when available. This is an exploratory stop/go rule, not statistical confirmation.',
          'new_independent_900_used':False,'evaluator_sha256':EVAL_SHA,'worker_sha256':sha(Path(__file__))}
    fixed(JOB/'plan.json',plan)
    if (DATA/'inputs.json').exists():
        inputs=read(DATA/'inputs.json');assert len(inputs)==300
        return inputs
    inputs=[];references={'calibration':{},'comparison':{}}
    from PIL import Image
    images=DATA/'images';images.mkdir(parents=True,exist_ok=True)
    for subject in subjects:
        metadata=[r for r in read(BASE_RESULTS/(subject+'.json')) if r['is_mcq']]
        selected=sorted(metadata,key=lambda r:hashlib.sha256(('soft-vote-dev-v1'+r['id']).encode()).hexdigest())[:10]
        assert len(selected)==10
        dataset=load_dataset('MMMU/MMMU',subject,split='validation',revision=plan['revision'])
        positions={candidate:i for i,candidate in enumerate(dataset['id'])}
        for index,item in enumerate(selected):
            row=dataset[positions[item['id']]]
            assert row['id'].startswith('validation_')
            options=ast.literal_eval(row['options']) if isinstance(row['options'],str) else row['options']
            assert 2<=len(options)<=16
            phase='calibration' if index%2==0 else 'comparison'
            record={'id':row['id'],'subject':subject,'question':row['question'],'options':row['options'],
                    'phase':phase,'option_count':len(options),'images':{},'image_sha256':{}}
            for n in range(1,8):
                image=row.get(f'image_{n}')
                if image is not None:
                    path=images/(row['id']+f'_image_{n}.png')
                    image.convert('RGB').save(path,format='PNG')
                    record['images'][f'image_{n}']=str(path.relative_to(PROJECT))
                    record['image_sha256'][f'image_{n}']=sha(path)
            record['input_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            inputs.append(record)
            assert row['answer'] in [chr(65+i) for i in range(len(options))]
            references[phase][row['id']]=row['answer']
        print(subject,': 개발 문항 10개 준비 완료',flush=True)
    assert len(inputs)==300 and all(len(v)==150 for v in references.values())
    save(DATA/'calibration_reference.json',references['calibration'])
    save(DATA/'comparison_reference.json',references['comparison'])
    save(DATA/'inputs.json',inputs)
    return inputs

def option_probs(logprobs,groups):
    values=[]
    for letter,ids in groups.items():
        assert all(token in logprobs for token in ids),'선택지 토큰 확률이 빠졌습니다.'
        entries=[float(logprobs[token].logprob) for token in ids]
        assert all(math.isfinite(x) for x in entries)
        values.append(sum(math.exp(x) for x in entries))
    mass=sum(values)
    assert abs(mass-1.0)<1e-4,f'조건부 확률 합이 1이 아닙니다: {mass}. 결과를 사용하지 않습니다.'
    return [x/mass for x in values]

def worker(name,deadline):
    helper=helper_module();helper.configure()
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    rows=read(DATA/'inputs.json');path=JOB/'probabilities'/(name+'.json')
    values=read(path) if path.exists() else []
    assert len(values)<=300
    for value,row in zip(values,rows):assert value['id']==row['id'] and value['input_sha256']==row['input_sha256']
    if len(values)==300:return
    if time.time()>=deadline:raise SystemExit(75)
    model=LLM(model=str(MODELS[name]),max_model_len=4096,gpu_memory_utilization=0.85,
              seed=3407,trust_remote_code=True,enforce_eager=True,max_num_seqs=4,
              limit_mm_per_prompt={'image':8,'video':0},max_logprobs=64,logprobs_mode='processed_logprobs')
    tokenizer=model.get_tokenizer()
    processor=AutoProcessor.from_pretrained(MODELS['A'],local_files_only=True)
    functions=helper.primary_functions()
    mappings={}
    for count in {r['option_count'] for r in rows}:
        mapping={}
        for i in range(count):
            letter=chr(65+i);ids=set()
            for text in [letter,' '+letter]:
                encoded=tokenizer.encode(text,add_special_tokens=False)
                if len(encoded)==1:ids.add(encoded[0])
            assert ids,f'선택지 토큰을 확인하지 못했습니다: {letter}'
            mapping[letter]=sorted(ids)
        assert sum(len(v) for v in mapping.values())==len({i for v in mapping.values() for i in v})
        mappings[count]=mapping
    fixed(JOB/'token_mappings.json',{str(k):v for k,v in mappings.items()})
    started=time.monotonic();initial=len(values)
    for start in range(initial,300,10):
        if time.time()>=deadline:raise SystemExit(75)
        batch=rows[start:start+10];requests=[];parameters=[];owners=[]
        for row in batch:
            value,letters,is_mcq,owned=helper.request(row,processor,functions,answer_first=True)
            assert is_mcq and len(letters)==row['option_count']
            value['prompt']+='Answer:'
            mapping=mappings[row['option_count']]
            allowed=[token for ids in mapping.values() for token in ids]
            requests.append(value);owners.extend(owned)
            parameters.append(SamplingParams(temperature=0.0,top_p=1.0,top_k=-1,max_tokens=1,
                              repetition_penalty=1.0,presence_penalty=0.0,allowed_token_ids=allowed,
                              logprobs=len(allowed),seed=3407))
        results=model.generate(requests,parameters,use_tqdm=False)
        for row,result in zip(batch,results):
            logs=result.outputs[0].logprobs
            assert logs and len(logs)==1
            probabilities=option_probs(logs[0],mappings[row['option_count']])
            values.append({'id':row['id'],'input_sha256':row['input_sha256'],'phase':row['phase'],
                           'probabilities':probabilities,'predicted':chr(65+max(range(len(probabilities)),key=lambda i:probabilities[i]))})
        helper.close_images(owners)
        save(path,values)
        elapsed=time.monotonic()-started;done=len(values)-initial
        seconds=elapsed/done*(300-len(values))
        status('선택지 확률 수집 중',모델=name,완료=len(values),전체=300,
               현재모델_예상남은초=round(seconds),실행예산_남은초=max(0,round(deadline-time.time())),로그=str(JOB/(name+'.log')))
        print(name,':',len(values),'/300, 현재 모델 예상 남은 시간',round(seconds),'초',flush=True)

def tempered(values,t):
    logits=[math.log(max(x,1e-30))/t for x in values];peak=max(logits)
    exp=[math.exp(v-peak) for v in logits];total=sum(exp)
    return [v/total for v in exp]
def nll(predictions,labels):
    return -sum(math.log(max(predictions[k][ord(gold)-65],1e-30)) for k,gold in labels.items())/len(labels)
def soft(name,probabilities,temps):
    group='three' if name.endswith('three') else 'five'
    result={}
    for candidate in next(iter(probabilities.values())):
        vectors=[tempered(probabilities[m][candidate],temps.get(m,1.0)) for m in GROUPS[group]]
        result[candidate]=[sum(v[i] for v in vectors)/len(vectors) for i in range(len(vectors[0]))]
    return result

def compare():
    from frozen_ensemble_rule import choose_answer
    rows=read(DATA/'inputs.json');by_id={r['id']:r for r in rows}
    probabilities={}
    for name in MODELS:
        values=read(JOB/'probabilities'/(name+'.json'));assert len(values)==300
        assert all(v['id']==r['id'] and v['input_sha256']==r['input_sha256'] for v,r in zip(values,rows))
        probabilities[name]={v['id']:v['probabilities'] for v in values}
    calibration=read(DATA/'calibration_reference.json');assert len(calibration)==150
    temps={}
    for name,values in probabilities.items():
        losses={t:nll({k:tempered(p,t) for k,p in values.items()},calibration) for t in TEMPERATURES}
        temps[name]=min(TEMPERATURES,key=lambda t:(losses[t],abs(math.log(t))))
    soft_methods={name:soft(name,probabilities,temps if name.startswith('calibrated_') else {})
                  for name in ['soft_five','soft_three','calibrated_five','calibrated_three']}
    selected=min(soft_methods,key=lambda name:(nll(soft_methods[name],calibration),list(soft_methods).index(name)))
    best_individual=max(MODELS,key=lambda name:(sum(chr(65+max(range(len(probabilities[name][k])),key=lambda i:probabilities[name][k][i]))==gold for k,gold in calibration.items()),-list(MODELS).index(name)))
    fixed(JOB/'calibration_selection.json',{'temperatures':temps,'selected_soft_method':selected,
          'selected_individual':best_individual,'calibration_nll':{name:nll(values,calibration) for name,values in soft_methods.items()},
          'rule':'Temperature and method selection use calibration labels only. Comparison labels are loaded after this file is fixed.'})
    comparison=read(DATA/'comparison_reference.json');assert len(comparison)==150 and not set(calibration)&set(comparison)
    predictions={name:{k:chr(65+max(range(len(p)),key=lambda i:p[i])) for k,p in values.items()} for name,values in probabilities.items()}
    for group,names in GROUPS.items():
        predictions['hard_'+group]={k:choose_answer({name:predictions[name][k] for name in names},'three_models' if group=='three' else 'five_models')[0] for k in by_id}
    for name,values in soft_methods.items():
        predictions[name]={k:chr(65+max(range(len(p)),key=lambda i:p[i])) for k,p in values.items()}
    old=PROJECT/'analysis_out/ensemble_exploration_20261121/five_models_predictions.json'
    if old.is_file():
        previous={r['id']:r['predicted'] for r in read(old)}
        assert set(by_id)<=set(previous)
        predictions['old_generative_five_reference']={k:previous[k] for k in by_id}
    result=[]
    for name,values in predictions.items():
        result.append({'방법':name,'보정 150개 정답':sum(values[k]==gold for k,gold in calibration.items()),
                       '비교 150개 정답':sum(values[k]==gold for k,gold in comparison.items())})
    scores={r['방법']:r['비교 150개 정답'] for r in result}
    gain=scores[selected]-scores['hard_five']
    old_gain=scores[selected]-scores['old_generative_five_reference'] if 'old_generative_five_reference' in scores else None
    report={'결과':result,'보정으로 선정한 확률 결합':selected,'모델별 보정 온도':temps,
            '선정 확률 결합의 단순 5모델 투표 대비 순증가':gain,
            '기존 자유 생성 5모델 참고값 대비 순증가':old_gain,
            '강한 후속 실험 신호':gain>=5 and scores[selected]>=scores[best_individual] and (old_gain is None or old_gain>=3),
            '주의':'이미 여러 개발 실험에 사용한 기존 Validation의 작은 개발 비교입니다. 새 독립 검증 결과가 아닙니다. 직접 선택지 채점과 기존 자유 생성의 프롬프트 및 출력 방식이 다릅니다.',
            '다음 단계':'개선 신호가 있으면 방법을 고정하고 아직 사용하지 않은 새 Test 문항에서 별도 검증. 기존 독립 900문항으로 다시 선택하지 않음.'}
    save(JOB/'comparison.json',report)
    status('확률 앙상블 개발 비교 완료',결과=result,선정방법=selected,단순투표대비증가=gain,결과파일=str(JOB/'comparison.json'))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def run(minutes):
    import fcntl
    helper=helper_module();helper.configure()
    os.environ['HF_DATASETS_OFFLINE']='1'
    with (JOB/'pilot.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('같은 확률 실험이 이미 실행 중입니다.')
        deadline=time.time()+minutes*60
        try:
            used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            assert int(used.splitlines()[0])<2000,'다른 GPU 작업이 실행 중일 수 있습니다. 동시에 시작하지 않습니다.'
            status('기존 Validation 개발 문항 준비 중')
            prepare()
            for name in MODELS:
                path=JOB/'probabilities'/(name+'.json')
                if path.exists() and len(read(path))==300:continue
                remaining=deadline-time.time()
                if remaining<=0:
                    status('시간 예산 도달. 개발 비교 미완료',남은모델=name);return
                logpath=JOB/(name+'.log')
                status('선택지 확률 모델 로딩 중',모델=name,로그=str(logpath),예산_분=minutes)
                with logpath.open('a',encoding='utf-8') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name,'--deadline',str(deadline)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=process.wait(timeout=max(1,deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
                        status('시간 예산 도달. 개발 비교 미완료',현재모델=name,저장확률=str(JOB/'probabilities'));return
                if code==75:
                    status('시간 예산 도달. 개발 비교 미완료',현재모델=name,저장확률=str(JOB/'probabilities'));return
                assert code==0,f'{name} 확률 수집 실패. 로그: {logpath}'
            compare()
        except Exception:
            status('오류로 중단',오류=traceback.format_exc());raise

def launch(minutes):
    JOB.mkdir(parents=True,exist_ok=True)
    target=JOB/'run.py';source=Path(__file__).read_bytes()
    if target.exists():assert target.read_bytes()==source,'저장된 실행 코드와 다릅니다.'
    else:target.write_bytes(source)
    with (JOB/'runner.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',str(target),'--run','--minutes',str(minutes)],cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(JOB/'launch.json',{'pid':process.pid,'budget_minutes':minutes})
    print('확률 앙상블 개발 실험 실행 요청 완료')
    print('실행 예산:',minutes,'분')
    print('프로세스 번호:',process.pid)
    print('로그:',JOB/'runner.log')

def monitor(watch):
    while True:
        state=read(JOB/'status.json') if (JOB/'status.json').exists() else {'상태':'아직 상태 파일 없음'}
        print('\n확인 시각:',datetime.now().isoformat(timespec='seconds'))
        print(json.dumps(state,ensure_ascii=False,indent=2))
        for name in MODELS:
            p=JOB/'probabilities'/(name+'.json');print(name,':',len(read(p)) if p.exists() else 0,'/300개 저장')
        log=Path(state.get('로그',str(JOB/'runner.log')))
        if log.exists():print('최근 로그:',log,'\n'+'\n'.join(log.read_text(encoding='utf-8',errors='replace').replace('\r','\n').splitlines()[-8:]))
        if not watch or state['상태'] in ['확률 앙상블 개발 비교 완료','오류로 중단','시간 예산 도달. 개발 비교 미완료']:break
        time.sleep(10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',choices=list(MODELS));parser.add_argument('--deadline',type=float)
    parser.add_argument('--minutes',type=int,default=55);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    assert 1<=args.minutes<=55,'이번 예비 실험의 실행 예산은 최대 55분입니다.'
    if args.launch:launch(args.minutes)
    elif args.run:run(args.minutes)
    elif args.worker:worker(args.worker,args.deadline)
    else:
        try:monitor(args.watch)
        except KeyboardInterrupt:print('\n확인 화면만 종료했습니다. 실험은 계속됩니다.')
