from pathlib import Path
from collections import Counter
import argparse
import ast
import importlib.util
import json
import math
import re
import sys
import os
import subprocess
import fcntl
import unicodedata

PROJECT=Path('/mnt/c/Users/K/Desktop/MMDL')
JOB=PROJECT/'logs/mmmu_pro_final_20261121'
RESULTS=PROJECT/'results/mmmu_pro_final_20261121'
ORIGINAL=PROJECT/'results/mmmu_pro_base_R20_20261121'
sys.path.insert(0,str(PROJECT/'scripts'))
spec=importlib.util.spec_from_file_location('pro_reference',PROJECT/'scripts/run_mmmu_pro.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.JOB=JOB;g.RESULTS=RESULTS;g.__file__=str(Path(__file__).resolve())
g.MODELS={
 'base':g.MODELS['base'],
 'A':PROJECT/'merged_models/recipe_a_lr_2e-4_step500_20261116_232956_544892',
 'R20':g.MODELS['R20'],
 'hard_answer':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_answer',
 'hard_reasoning':PROJECT/'merged_models/mmmu_hard_20261121_043545_392882/hard_reasoning',
}


def setup_parser():
    evaluator=PROJECT/'scripts/eval_mmmu_vllm.py'
    assert g.digest(evaluator)=='0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
    def unwrap(value):
        value=value.strip()
        for wrapper in ('**','__','`'):
            if value.startswith(wrapper) and value.endswith(wrapper) and len(value)>2*len(wrapper):
                value=value[len(wrapper):-len(wrapper)].strip()
        return value
    def parse(raw,count):
        assert 2<=count<=26
        text=unwrap(raw)
        explicit=re.findall(r'^\s*(?:\*\*)?Answer\s*:\s*(.*?)\s*$',text,flags=re.I|re.M)
        candidate=unwrap(explicit[-1]) if explicit else text
        match=re.fullmatch(r'(?:\(([A-Z])\)|([A-Z])(?:[.):\]]|(?=\s|$)))\s*(.*)',candidate,flags=re.I|re.S)
        if not match:return None
        letter=(match.group(1) or match.group(2)).upper()
        return letter if 0<=ord(letter)-65<count else None
    g.parse_answer=parse
    g.PROTOCOL={**g.PROTOCOL,
      'grading':'Uniform audited letter parser: last explicit Answer line, or a response-leading valid letter with optional option text. No search for arbitrary letters in free explanation, no random fallback and no retry. Same parser for all five models; saved base/R20 outputs are reparsed without regeneration.',
      'grading_source_sha256':g.digest(Path(__file__)),
      'original_validation_evaluator_sha256':g.digest(evaluator),
      'reference_runner_sha256':g.digest(PROJECT/'scripts/run_mmmu_pro.py'),
      'voting_code_sha256':g.digest(PROJECT/'scripts/frozen_ensemble_rule.py'),
      'reuse':'Original base and R20 raw generation outputs are reused. Only A, hard_answer and hard_reasoning require new inference.',
      'interpretation':'Supplementary comparison on already observed MMMU-Pro. This is separate from the independent-900 result, not another untouched holdout.'}


def seed_existing_predictions():
    for setting in g.CONFIGS:
        for name in ['base','R20']:
            source=ORIGINAL/name/(setting+'_predictions.json')
            target=RESULTS/name/(setting+'_predictions.json')
            rows=g.read(source)
            assert len(rows)==1730 and len({row['id'] for row in rows})==1730
            converted=[]
            for row in rows:
                predicted=g.parse_answer(row['raw'],row['option_count'])
                converted.append({**row,'predicted':predicted,'correct':predicted==row['gold']})
            if target.exists():assert g.read(target)==converted
            else:g.save(target,converted)


def run_final():
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'preparation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('최종 MMMU-Pro 평가가 이미 실행 중입니다.')
        try:
            used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            assert used<2000,'다른 GPU 작업이 실행 중일 수 있습니다. 동시에 실행하지 않습니다.'
            g.status('최종 Pro 모델 파일 검증 중')
            binary=g.read(PROJECT/'logs/independent_900_20261121/model_binary_lock.json')
            for name,folder in g.MODELS.items():
                assert str(folder)==binary[name]['path']
                assert g.digest(folder/'config.json')==binary[name]['config_sha256']
                for filename,expected in binary[name]['weights'].items():
                    assert g.digest(folder/filename)==expected
            seed_existing_predictions()
            g.run()
        except Exception:
            import traceback
            g.status('오류로 중단',오류=traceback.format_exc())
            raise


def dispatch_worker(name,setting):
    assert name in g.MODELS and setting in g.CONFIGS
    path=RESULTS/name/(setting+'_predictions.json')
    if name in ('base','R20') and path.exists():
        rows=g.read(path);bindings=g.read(JOB/(setting+'_inputs.json'))
        assert len(rows)==len(bindings)==1730
        assert all(all(row[key]==binding[key] for key in ('id','subject','gold','option_count','input_sha256')) for row,binding in zip(rows,bindings))
        assert all(row['predicted']==g.parse_answer(row['raw'],row['option_count']) and row['correct']==(row['predicted']==row['gold']) for row in rows)
        g.save(path.parent/(setting+'_summary.json'),g.summary(rows))
        print(name,setting,': 기존 원시 출력1730개를 같은 파서로 검사했습니다. 새 GPU 추론은 생략합니다.',flush=True)
        return
    g.worker(name,setting)


def paired(left,right):
    recovered=sum(not a and b for a,b in zip(left,right));regressed=sum(a and not b for a,b in zip(left,right));n=recovered+regressed
    p=min(1,2*sum(math.comb(n,k) for k in range(min(recovered,regressed)+1))/(2**n)) if n else 1
    return {'회복':recovered,'퇴보':regressed,'순증가':recovered-regressed,'McNemar exact p':p}


def compare():
    from frozen_ensemble_rule import choose_answer,GROUPS
    protocol=g.read(PROJECT/'evaluation_reference/independent_900_20261121/candidate_protocol.json')
    assert g.digest(PROJECT/'scripts/frozen_ensemble_rule.py')==protocol['voting_code_sha256']
    report={}
    for setting in g.CONFIGS:
        models={name:g.read(RESULTS/name/(setting+'_predictions.json')) for name in g.MODELS}
        reference=models['base']
        assert len(reference)==1730
        for name,rows in models.items():
            assert len(rows)==1730
            assert all(all(row[k]==ref[k] for k in ['id','gold','option_count','input_sha256']) for row,ref in zip(rows,reference))
            assert all(row['predicted']==g.parse_answer(row['raw'],row['option_count']) for row in rows)
        for group,names in GROUPS.items():
            votes=[]
            for index,ref in enumerate(reference):
                answer,selected=choose_answer({name:models[name][index]['predicted'] for name in names},group)
                votes.append({'id':ref['id'],'predicted':answer,'selected_model':selected,'input_sha256':ref['input_sha256']})
            g.save(RESULTS/'blind_votes'/(setting+'_'+group+'.json'),votes)
            models[group]=[{**models[vote['selected_model']][index],'predicted':vote['predicted'],'correct':vote['predicted']==ref['gold']} for index,(ref,vote) in enumerate(zip(reference,votes))]
        scores={name:g.summary(rows) for name,rows in models.items()}
        for name,rows in models.items():
            scores[name]['순정 대비']=paired([r['correct'] for r in reference],[r['correct'] for r in rows])
            scores[name]['R20 대비']=paired([r['correct'] for r in models['R20']],[r['correct'] for r in rows])
        report[setting]=scores
    g.save(RESULTS/'comparison.json',{'결과':report,'protocol':g.read(JOB/'protocol.json'),
           '주의':'기존 Pro 출력은 보존했습니다. 전 모델에 같은 답 추출 규칙을 적용한 추가 비교입니다. 독립 900문항과 구분하며 전체 MMMU Test 점수로 표시하지 않습니다.'})
    g.status('완료',단계='최종 후보 Pro 평가와 고정 투표 완료',결과={setting:{name:value['correct'] for name,value in report[setting].items()} for setting in report},결과파일=str(RESULTS/'comparison.json'))
    print(json.dumps(g.read(JOB/'status.json'),ensure_ascii=False,indent=2),flush=True)


def progress_print(*args,**kwargs):
    values=[re.sub(r', 정답 \d+','',value) if isinstance(value,str) else value for value in args]
    print(*values,**kwargs)

setup_parser()
g.compare=compare
g.print=progress_print

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',nargs=2);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        previous=JOB/'launch.json'
        if previous.exists():
            info=g.read(previous);command=Path('/proc')/str(info['pid'])/'cmdline'
            assert not (command.exists() and str(Path(__file__).resolve()).encode() in command.read_bytes().split(b'\0')),'이미 실행 중입니다. 확인 명령만 실행하세요.'
        g.launch()
    elif args.run:
        run_final()
    elif args.worker:dispatch_worker(*args.worker)
    else:
        try:g.monitor(args.watch)
        except KeyboardInterrupt:print('\n확인 화면만 종료했습니다. 백그라운드 평가는 계속됩니다.')
