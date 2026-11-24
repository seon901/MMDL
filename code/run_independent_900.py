from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

PROJECT=Path('/mnt/c/Users/K/Desktop/MMDL')
DATA=PROJECT/'evaluation_reference/independent_900_20261121'
JOB=PROJECT/'logs/independent_900_20261121'
RESULTS=PROJECT/'results/independent_900_20261121'
EVALUATOR=PROJECT/'scripts/eval_mmmu_vllm.py'
EVALUATOR_SHA='0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
INPUT_SHA='44e98635286cec028570bb349775c0035186f96c8883c5fc4fc9d77f43fb28c1'
SAMPLE_SHA='5512396a637c2a519c079bc78f8406c63fe7efef18afec9347d185bf97baaea9'
BASE=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17')
SETTINGS=PROJECT/'merged_models/recipe_a_lr_2e-4_step500_20261116_232956_544892'
MIX_ADAPTER=PROJECT/'eval_candidates/independent_900_20261121/mix_90_10'
MODELS={
    'base':BASE,
    'A':SETTINGS,
    'R20':PROJECT/'merged_models/mmmu_candidates_20261120_004215_155629/R20',
    'hard_answer':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_answer',
    'hard_reasoning':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_reasoning',
    'mix_90_10':PROJECT/'merged_models/independent_900_20261121/mix_90_10',
}
sys.path.insert(0,str(PROJECT/'scripts'))

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def fixed(path,value):
    if path.exists():assert read(path)==value,f'고정 기록과 다릅니다: {path}'
    else:save(path,value)
def status(state,**extra):save(JOB/'status.json',{'상태':state,'갱신 시각':datetime.now().isoformat(),**extra})
def configure():
    from resume_mmmu_candidates import configure as cuda_configure
    cuda_configure()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def primary_functions(grading=False):
    import re
    assert sha(EVALUATOR)==EVALUATOR_SHA
    wanted={'normalize_str','grade_open'} if grading else {'build_prompt_and_images','extract_mcq_answer','extract_open_answer','build_vllm_input'}
    tree=ast.parse(EVALUATOR.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in functions}==wanted
    namespace={'re':re,'ast':ast,'MIN_PIXELS':200704,'MAX_PIXELS':802816,'LETTER_POOL':'ABCDEFGHIJKLMNOP'}
    if not grading:
        from qwen_vl_utils import process_vision_info
        namespace['process_vision_info']=process_vision_info
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(EVALUATOR),'exec'),namespace)
    return namespace

def blind_rows():
    assert sha(DATA/'blind_inputs.json')==INPUT_SHA
    assert sha(DATA/'screened_sample_manifest.json')==SAMPLE_SHA
    rows=read(DATA/'blind_inputs.json')
    assert len(rows)==900 and len({r['id'] for r in rows})==900
    counts=Counter(r['subject'] for r in rows)
    assert len(counts)==30 and set(counts.values())=={30}
    forbidden={'answer','ground_truth','gold','correct','explanation'}
    assert all(not forbidden&set(r) for r in rows)
    return rows

def sample(row):
    from PIL import Image
    result={'id':row['id'],'question':row['question'],'options':row['options']}
    owners=[]
    for name,relative in row['images'].items():
        path=PROJECT/relative
        assert sha(path)==row['image_sha256'][name]
        with Image.open(path) as opened:image=opened.convert('RGB')
        result[name]=image;owners.append(image)
    return result,owners

def request(row,processor,functions,answer_first=False):
    data,owners=sample(row)
    value,letters,is_mcq=functions['build_vllm_input'](data,processor,answer_first)
    owners.extend(value.get('multi_modal_data',{}).get('image',[]))
    return value,letters,is_mcq,owners

def close_images(owners):
    for image in owners:image.close()

def make_mix():
    import numpy as np
    from safetensors.numpy import load_file,save_file
    protocol=read(DATA/'candidate_protocol.json')
    names=['R20','hard_reasoning'];weights=[0.9,0.1]
    sources={name:PROJECT/protocol['adapters'][name]['directory'] for name in names}
    identity={'source_sha256':{name:protocol['adapters'][name]['weights_sha256'] for name in names},'weights':weights,'r':16,'alpha':16}
    if (MIX_ADAPTER/'mix_record.json').is_file():
        record=read(MIX_ADAPTER/'mix_record.json')
        assert record['identity']==identity and sha(MIX_ADAPTER/'adapter_model.safetensors')==record['adapter_sha256']
        return
    configs={name:read(folder/'adapter_config.json') for name,folder in sources.items()}
    tensors={}
    for name,folder in sources.items():
        assert sha(folder/'adapter_model.safetensors')==identity['source_sha256'][name]
        assert configs[name]['r']==8 and configs[name]['lora_alpha']==16
        assert not configs[name].get('use_rslora') and not configs[name].get('use_dora')
        tensors[name]=load_file(str(folder/'adapter_model.safetensors'))
    assert set(tensors[names[0]])==set(tensors[names[1]])
    keys=sorted(k for k in tensors[names[0]] if k.endswith('.lora_A.weight'))
    assert len(keys)==144 and len(tensors[names[0]])==288
    output={};rng=np.random.default_rng(42)
    for key in keys:
        bkey=key.replace('.lora_A.weight','.lora_B.weight')
        a0,a1=tensors[names[0]][key],tensors[names[1]][key]
        b0,b1=tensors[names[0]][bkey],tensors[names[1]][bkey]
        a=np.concatenate([a0*(weights[0]*2),a1*(weights[1]*2)],axis=0)
        b=np.concatenate([b0,b1],axis=1)
        probe=rng.normal(size=(a0.shape[1],2)).astype(np.float32)
        expected=weights[0]*2*(b0@(a0@probe))+weights[1]*2*(b1@(a1@probe))
        assert np.allclose(b@(a@probe),expected,rtol=1e-4,atol=1e-5)
        output[key]=np.ascontiguousarray(a);output[bkey]=np.ascontiguousarray(b)
    MIX_ADAPTER.parent.mkdir(parents=True,exist_ok=True)
    assert not MIX_ADAPTER.exists(),'혼합 어댑터 폴더가 불완전합니다. 기존 파일을 확인해야 합니다.'
    stage=Path(tempfile.mkdtemp(prefix='mix_build_',dir=MIX_ADAPTER.parent))
    config=copy.deepcopy(configs[names[0]]);config.update(r=16,lora_alpha=16,inference_mode=True)
    save(stage/'adapter_config.json',config)
    save_file(output,str(stage/'adapter_model.safetensors'),metadata={'format':'pt'})
    save(stage/'mix_record.json',{'identity':identity,'verified_modules':144,'adapter_sha256':sha(stage/'adapter_model.safetensors')})
    stage.rename(MIX_ADAPTER)

def merge_mix():
    configure()
    dependencies=PROJECT/'merge_dependencies_Sb7GdJ'
    assert dependencies.is_dir()
    sys.path.insert(0,str(dependencies))
    import torch,peft,accelerate
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from transformers import Qwen3VLForConditionalGeneration
    assert peft.__version__=='0.17.1' and accelerate.__version__=='1.7.0'
    assert Path(peft.__file__).resolve().is_relative_to(dependencies.resolve())
    destination=MODELS['mix_90_10']
    assert not destination.exists()
    assert shutil.disk_usage(PROJECT).free>=12*1024**3
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='mix_merge_',dir=destination.parent))
    model=Qwen3VLForConditionalGeneration.from_pretrained(BASE,dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True)
    model=PeftModel.from_pretrained(model,MIX_ADAPTER,is_trainable=False,local_files_only=True)
    assert model.peft_config['default'].r==16 and model.peft_config['default'].lora_alpha==16
    layers=[layer for layer in model.modules() if isinstance(layer,LoraLayer) and 'default' in layer.scaling]
    assert len(layers)==144 and all(abs(layer.scaling['default']-1.0)<1e-9 for layer in layers)
    model.merge_and_unload(safe_merge=True).save_pretrained(stage,safe_serialization=True,max_shard_size='4GB')
    for path in SETTINGS.iterdir():
        if path.is_file() and path.suffix in {'.json','.txt','.jinja','.model','.tiktoken'} and path.name not in {'merge_record.json','model.safetensors.index.json'}:
            shutil.copy2(path,stage/path.name)
    index=read(stage/'model.safetensors.index.json')
    assert set(index['weight_map'])==set(read(BASE/'model.safetensors.index.json')['weight_map'])
    save(stage/'merge_record.json',{'source_adapter_sha256':sha(MIX_ADAPTER/'adapter_model.safetensors'),
         'source_mix_record':read(MIX_ADAPTER/'mix_record.json'),'weights_sha256':{name:sha(stage/name) for name in sorted(set(index['weight_map'].values()))}})
    stage.rename(destination)

def preflight():
    from transformers import AutoProcessor
    protocol=read(DATA/'candidate_protocol.json')
    assert sha(PROJECT/'scripts/frozen_ensemble_rule.py')==protocol['voting_code_sha256']
    for name,item in protocol['adapters'].items():
        folder=PROJECT/item['directory']
        assert sha(folder/'adapter_model.safetensors')==item['weights_sha256']
        assert sha(folder/'adapter_config.json')==item['config_sha256']
    processor=AutoProcessor.from_pretrained(SETTINGS,local_files_only=True)
    functions=primary_functions()
    maximum=0;maximum_id=None
    for index,row in enumerate(blind_rows()):
        for first in [False,True] if row['question_type']=='multiple-choice' else [False]:
            value,_,_,owners=request(row,processor,functions,first)
            images=value.get('multi_modal_data',{}).get('image')
            kwargs={'text':[value['prompt']],'return_tensors':'pt'}
            if images:kwargs['images']=images
            encoded=processor(**kwargs)
            length=int(encoded['input_ids'].shape[-1])
            if length>maximum:maximum=length;maximum_id=row['id']
            close_images(owners)
        if (index+1)%100==0:
            status('정답 없는 입력 사전 검사 중',완료=index+1,전체=900)
            print('입력 사전 검사:',index+1,'/900',flush=True)
    context=max(4096,math.ceil((maximum+512)/1024)*1024)
    assert context<=8192,'새 표본의 입력 길이가 예상 범위를 초과했습니다. 모델 추론을 시작하지 않습니다.'
    execution={'candidate_protocol_sha256':sha(DATA/'candidate_protocol.json'),
               'sample_manifest_sha256':SAMPLE_SHA,'blind_inputs_sha256':INPUT_SHA,
               'grading_reference_sha256':sha(DATA/'grading_reference_screened.json'),
               'voting_code_sha256':protocol['voting_code_sha256'],'worker_code_sha256':sha(Path(__file__)),
               'original_evaluator_sha256':EVALUATOR_SHA,'max_prompt_tokens':maximum,'longest_input_id':maximum_id,
               'max_model_len':context,'context_rule':'Before any model outputs, reserve 512 output tokens and round the observed maximum prompt length to a 1024-token boundary, minimum 4096 and maximum 8192. Use the same context limit for every candidate.',
               'models':{name:str(path) for name,path in MODELS.items()},
               'generation':protocol['inference'],'scores_hidden_until_all_predictions_complete':True}
    fixed(JOB/'execution_lock.json',execution)
    return execution

def worker(name):
    configure()
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    functions=primary_functions()
    execution=read(JOB/'execution_lock.json')
    assert sha(Path(__file__))==execution['worker_code_sha256']
    rows=blind_rows();subjects=sorted({r['subject'] for r in rows})
    output=RESULTS/'blind_predictions'/name;output.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(SETTINGS,local_files_only=True)
    completed=[]
    for subject in subjects:
        path=output/(subject+'.json')
        if path.exists():
            records=read(path)
            wanted=[r for r in rows if r['subject']==subject]
            assert len(records)==30 and all(a['id']==b['id'] and a['input_binding']==b['input_text_sha256'] for a,b in zip(records,wanted))
            completed.extend(records)
    if len(completed)==900:
        save(output/'all_predictions.json',completed);return
    llm=LLM(model=str(MODELS[name]),limit_mm_per_prompt={'image':8,'video':0},max_model_len=execution['max_model_len'],
            gpu_memory_utilization=0.90,seed=3407,trust_remote_code=True,enforce_eager=True,max_num_seqs=4)
    parameters=SamplingParams(temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0,presence_penalty=1.5,max_tokens=512,seed=3407)
    for subject in subjects:
        path=output/(subject+'.json')
        if path.exists():continue
        selected=[r for r in rows if r['subject']==subject]
        requests=[];owners=[];letters=[];mcq=[]
        for row in selected:
            value,letter,is_mcq,owned=request(row,processor,functions)
            requests.append(value);owners.extend(owned);letters.append(letter);mcq.append(is_mcq)
        status('정답을 읽지 않는 모델 추론 중',모델=name,과목=subject,완료=len(completed),전체=900,로그=str(JOB/(name+'.log')))
        outputs=llm.generate(requests,parameters,use_tqdm=False)
        records=[]
        for row,result,letter,is_mcq in zip(selected,outputs,letters,mcq):
            text=result.outputs[0].text
            predicted=functions['extract_mcq_answer'](text,letter) if is_mcq else functions['extract_open_answer'](text)
            records.append({'id':row['id'],'subject':subject,'is_mcq':is_mcq,'predicted':predicted,
                            'raw_output':text,'first_raw_output':text,'retried':False,
                            'finish_reason':result.outputs[0].finish_reason,'input_binding':row['input_text_sha256']})
        retry=[i for i,row in enumerate(records) if row['is_mcq'] and row['predicted'] is None]
        if retry:
            retry_requests=[]
            for i in retry:
                value,_,_,owned=request(selected[i],processor,functions,True)
                retry_requests.append(value);owners.extend(owned)
            for i,result in zip(retry,llm.generate(retry_requests,parameters,use_tqdm=False)):
                text=result.outputs[0].text
                records[i].update(predicted=functions['extract_mcq_answer'](text,letters[i]),raw_output=text,
                                  retried=True,finish_reason=result.outputs[0].finish_reason)
        close_images(owners)
        save(path,records);completed.extend(records)
        print(name,subject,':',len(completed),'/900개 예측 저장. 채점하지 않았습니다.',flush=True)
    assert len(completed)==900
    save(output/'all_predictions.json',completed)

def pair(before,after):
    a=sum(not x and y for x,y in zip(before,after));b=sum(x and not y for x,y in zip(before,after));n=a+b
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n)) if n else 1.0
    return {'회복':a,'퇴보':b,'순증가':a-b,'McNemar exact p':p}

def score_after_sealing():
    from frozen_ensemble_rule import choose_answer,GROUPS
    execution=read(JOB/'execution_lock.json')
    assert sha(PROJECT/'scripts/frozen_ensemble_rule.py')==execution['voting_code_sha256']
    rows=blind_rows();ids=[r['id'] for r in rows]
    predictions={};files={}
    for name in MODELS:
        path=RESULTS/'blind_predictions'/name/'all_predictions.json'
        values=read(path)
        assert len(values)==900 and {r['id'] for r in values}==set(ids)
        assert all(not {'answer','correct','gold','ground_truth'}&set(r) for r in values)
        predictions[name]={r['id']:r for r in values};files[str(path.relative_to(PROJECT))]=sha(path)
    for group,names in GROUPS.items():
        values=[]
        for candidate in ids:
            answers={name:predictions[name][candidate]['predicted'] for name in names}
            answer,selected=choose_answer(answers,group)
            values.append({'id':candidate,'predicted':answer,'selected_model':selected})
        path=RESULTS/'blind_votes'/(group+'.json');save(path,values)
        predictions[group]={r['id']:r for r in values};files[str(path.relative_to(PROJECT))]=sha(path)
    fixed(RESULTS/'prediction_seal.json',{'sample_manifest_sha256':SAMPLE_SHA,'files':files,
          'rule':'Every planned model prediction and both gold-blind vote files are sealed before loading the grading reference.'})
    assert sha(DATA/'grading_reference_screened.json')==execution['grading_reference_sha256']
    reference=read(DATA/'grading_reference_screened.json')
    assert set(reference)==set(ids)
    grader=primary_functions(grading=True)['grade_open']
    scored={};report={}
    by_id={r['id']:r for r in rows}
    for name,values in predictions.items():
        results=[]
        for candidate in ids:
            answer=values[candidate]['predicted'];gold=reference[candidate]['ground_truth']
            is_mcq=reference[candidate]['question_type']=='multiple-choice'
            correct=answer in (gold if isinstance(gold,list) else [gold]) if is_mcq else grader(gold,answer)
            results.append({'id':candidate,'subject':by_id[candidate]['subject'],'predicted':answer,'gold':gold,'correct':bool(correct)})
        scored[name]=results
        save(RESULTS/'scored'/(name+'.json'),results)
        counts=Counter(r['subject'] for r in results if r['correct'])
        report[name]={'correct':sum(r['correct'] for r in results),'total':900,
                      'accuracy':sum(r['correct'] for r in results)/900,
                      'subjects':{subject:{'correct':counts[subject],'total':30} for subject in sorted({r['subject'] for r in rows})}}
    for name,values in scored.items():
        report[name]['순정 대비']=pair([r['correct'] for r in scored['base']],[r['correct'] for r in values])
        report[name]['R20 대비']=pair([r['correct'] for r in scored['R20']],[r['correct'] for r in values])
    save(RESULTS/'comparison.json',{'결과':report,'execution_lock':execution,
         '구분':'새 Test 표본 900문항의 고정 후보 평가. 기존 Validation 탐색 결과 및 전체 Test 10500문항 점수와 구분.',
         '한계':'입력 유사도 검사는 보수적이며 모든 데이터 변형이나 사전학습 노출을 배제하지 못합니다. 앙상블과 단일 모델의 추론 횟수가 다릅니다. 여러 후보의 명목 p 값을 우월성 확정으로 해석하지 않습니다.'})
    status('독립 900문항 평가와 고정 투표 및 채점 완료',결과={name:row['correct'] for name,row in report.items()},결과파일=str(RESULTS/'comparison.json'))
    print(json.dumps(read(JOB/'status.json'),ensure_ascii=False,indent=2),flush=True)

def run():
    import fcntl
    configure();JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'evaluation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('독립 평가 작업이 이미 실행 중입니다.')
        try:
            status('고정 후보와 입력 검사 중')
            for name,version in [('torch','2.8.0'),('transformers','4.57.6'),('vllm','0.11.0')]:
                assert importlib.metadata.version(name).split('+')[0]==version
            if not (JOB/'execution_lock.json').exists():preflight()
            else:
                execution=read(JOB/'execution_lock.json')
                assert execution['worker_code_sha256']==sha(Path(__file__))
                assert execution['blind_inputs_sha256']==sha(DATA/'blind_inputs.json')
                assert execution['candidate_protocol_sha256']==sha(DATA/'candidate_protocol.json')
                assert execution['voting_code_sha256']==sha(PROJECT/'scripts/frozen_ensemble_rule.py')
            make_mix()
            if not (MODELS['mix_90_10']/'merge_record.json').exists():
                status('고정 90대10 혼합 후보 병합 중',로그=str(JOB/'mix_merge.log'))
                with (JOB/'mix_merge.log').open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--merge-mix'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            model_lock={}
            for name,folder in MODELS.items():
                assert (folder/'config.json').is_file()
                index=read(folder/'model.safetensors.index.json')
                model_lock[name]={'path':str(folder),'config_sha256':sha(folder/'config.json'),'weights':{}}
                for filename in sorted(set(index['weight_map'].values())):
                    print('모델 파일 고정 검사:',name,filename,flush=True)
                    model_lock[name]['weights'][filename]=sha(folder/filename)
            fixed(JOB/'model_binary_lock.json',model_lock)
            for name in MODELS:
                logpath=JOB/(name+'.log')
                status('정답을 읽지 않는 모델 추론 중',모델=name,로그=str(logpath))
                with logpath.open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('모든 예측 완료. 고정 투표와 별도 채점 중')
            score_after_sealing()
        except Exception:
            status('오류로 중단',오류=traceback.format_exc());raise

def launch():
    configure();JOB.mkdir(parents=True,exist_ok=True)
    target=JOB/'run.py';source=Path(__file__).read_bytes()
    if target.exists():assert target.read_bytes()==source,'기존 실행 코드와 다릅니다.'
    else:target.write_bytes(source)
    with (JOB/'runner.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',str(target),'--run'],cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(JOB/'launch.json',{'pid':process.pid,'script':str(target)})
    print('독립 900문항 백그라운드 실행 요청 완료')
    print('프로세스 번호:',process.pid)
    print('로그:',JOB/'runner.log')

def monitor(watch):
    while True:
        state=read(JOB/'status.json') if (JOB/'status.json').is_file() else {'상태':'아직 상태 파일 없음'}
        print('\n확인 시각:',datetime.now().isoformat(timespec='seconds'))
        print(json.dumps(state,ensure_ascii=False,indent=2))
        launch_path=JOB/'launch.json'
        if launch_path.is_file():
            info=read(launch_path)
            command=Path('/proc')/str(info['pid'])/'cmdline'
            active=command.is_file() and str(JOB/'run.py').encode() in command.read_bytes()
            print('순차 실행 프로세스:', info['pid'], '실행 중' if active else '실행되지 않음')
            if not active and state['상태'] not in ['독립 900문항 평가와 고정 투표 및 채점 완료','오류로 중단']:
                print('작업이 중단됐을 수 있습니다. 로그와 하위 추론 프로세스를 확인해야 합니다.')
        for name in MODELS:
            folder=RESULTS/'blind_predictions'/name
            paths=list(folder.glob('*.json')) if folder.exists() else []
            count=sum(len(read(p)) for p in paths if p.name!='all_predictions.json')
            print(name,':',count,'/900개 저장')
        path=Path(state.get('로그',str(JOB/'runner.log')))
        if path.is_file():
            print('최근 로그:',path,'갱신',round(time.time()-path.stat().st_mtime),'초 전')
            print('\n'.join(path.read_text(encoding='utf-8',errors='replace').replace('\r','\n').splitlines()[-8:]))
        if state['상태']=='오류로 중단':
            print('실행 로그:',JOB/'runner.log')
        if not watch or state['상태'] in ['독립 900문항 평가와 고정 투표 및 채점 완료','오류로 중단']:break
        time.sleep(10)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--merge-mix',action='store_true');parser.add_argument('--worker',choices=list(MODELS))
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.launch:launch()
    elif args.run:run()
    elif args.merge_mix:merge_mix()
    elif args.worker:worker(args.worker)
    else:
        try:monitor(args.watch)
        except KeyboardInterrupt:print('\n확인 화면만 종료했습니다. 백그라운드 작업은 계속됩니다.')
