from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import unicodedata

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'analysis_out/ensemble_exploration_20261121'
FOLDERS = {
    'base': PROJECT / 'results_vllm_v2',
    'A': PROJECT / 'results/recipe_a_lr_2e-4_step500_retry_20261117_004730',
    'R20': PROJECT / 'results/mmmu_candidates_20261120_004215_155629/R20',
    'hard_answer': PROJECT / 'results/mmmu_hard_20261121_043545_392882/hard_answer',
    'hard_reasoning': PROJECT / 'results/mmmu_hard_20261121_043545_392882/hard_reasoning',
}
GROUPS = {
    'R20_A_base': ['R20','A','base'],
    'R20_A_reasoning': ['R20','A','hard_reasoning'],
    'five_models': ['R20','A','base','hard_answer','hard_reasoning'],
}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def key(row):
    value = row.get('predicted')
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split()) if value is not None and str(value).strip() else None

def paired(left,right):
    recovered=sum(not a and b for a,b in zip(left,right))
    regressed=sum(a and not b for a,b in zip(left,right))
    n=recovered+regressed
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(recovered,regressed)+1))/(2**n)) if n else 1.0
    return {'회복':recovered,'퇴보':regressed,'순증가':recovered-regressed,'McNemar exact p':p}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={}
    fingerprints={}
    expected=None
    for name,folder in FOLDERS.items():
        summary=read(folder/'_summary.json')
        settings={k:summary[k] for k in ['max_new_tokens','min_pixels','max_pixels','seed','presence_penalty','backend','prompt_style']}
        if expected is None: expected=settings
        assert settings==expected
        records={}
        for subject in summary['subject_accuracy']:
            path=folder/(subject+'.json')
            fingerprints[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for row in read(path):
                assert row['id'] not in records and row['subject']==subject
                records[row['id']]=row
        assert len(records)==900
        data[name]=records
    ids=sorted(data['R20'])
    for name in data:
        assert set(data[name])==set(ids)
        assert all(all(data[name][i][k]==data['R20'][i][k] for k in ['answer','subject','is_mcq']) for i in ids)
    report={}
    for label,names in GROUPS.items():
        rows=[]
        for candidate in ids:
            reference=data['R20'][candidate]
            votes=Counter(key(data[name][candidate]) for name in names if key(data[name][candidate]) is not None)
            selected='R20'
            if votes:
                top=max(votes.values())
                winners=[value for value,count in votes.items() if count==top]
                if top>=2 and len(winners)==1 and key(reference)!=winners[0]:
                    selected=next(name for name in names if key(data[name][candidate])==winners[0])
            chosen=data[selected][candidate]
            rows.append({'id':candidate,'subject':reference['subject'],'selected_model':selected,
                         'predicted':chosen['predicted'],'correct':chosen['correct'],'gold':reference['answer'],
                         'votes':dict(votes)})
        correct=[r['correct'] for r in rows]
        report[label]={'models':names,'correct':sum(correct),'total':900,
                       'R20 대비':paired([data['R20'][i]['correct'] for i in ids],correct),
                       '순정 대비':paired([data['base'][i]['correct'] for i in ids],correct),
                       '과목별 정답 수':dict(Counter(r['subject'] for r in rows if r['correct']))}
        save(OUT/(label+'_predictions.json'),rows)
    save(OUT/'comparison.json',{'결과':report,
         'rule':'Use the unique highest-vote answer only when it has at least two votes; otherwise use R20. Selection never uses gold labels or correctness. For five models this is a plurality rule, not necessarily an absolute majority.',
         'limitations':'Exploratory replay of already observed MMMU Validation predictions. Multiple combinations were inspected. Not an independent test or a single-model result. Live deployment requires inference from every model in the combination.',
         'settings':expected,'input_sha256':fingerprints})
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
