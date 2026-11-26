from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import urllib.request

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
OUT = PROJECT / 'evaluation_reference/independent_900_20261121'
REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
DATA = PROJECT / 'evaluation_reference/MMMU_Test' / REVISION
PRO = PROJECT / 'evaluation_reference/MMMU_Pro/0d7426df4ccb3d8704a992fe6850abee7c262127'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def fixed(path,value):
    if path.exists():assert read(path)==value, f'고정 기록과 다릅니다: {path}'
    else:save(path,value)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'MMDL-independent-evaluation/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response:return response.read()

def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi,snapshot_download
    assert PROJECT.is_dir()
    protocol=read(OUT/'candidate_protocol.json')
    assert protocol['evaluation_set']['sample_size']==900
    assert protocol['evaluation_set']['revision']==REVISION
    rule=PROJECT/'scripts/frozen_ensemble_rule.py'
    assert sha(rule)==protocol['voting_code_sha256']
    expected_subjects=set(read(PROJECT/'results_vllm_v2/_summary.json')['subject_accuracy'])
    assert len(expected_subjects)==30
    seen=set()
    for path in sorted((PRO/'standard (10 options)').glob('*.parquet')):
        seen.update(pq.read_table(path,columns=['id']).column('id').to_pylist())
    assert len(seen)==1730
    seen.update(row['id'] for subject in expected_subjects for row in read(PROJECT/'results_vllm_v2'/(subject+'.json')))
    repo_files=list(HfApi().list_repo_tree('MMMU/MMMU',repo_type='dataset',revision=REVISION,recursive=True))
    files=[entry for entry in repo_files if hasattr(entry,'size') and re.fullmatch(r'[^/]+/test-\d+-of-\d+\.parquet',entry.path) and entry.path.split('/')[0] in expected_subjects]
    assert {entry.path.split('/')[0] for entry in files}==expected_subjects
    data_lock={'repository':'MMMU/MMMU','revision':REVISION,'files':{entry.path:{'bytes':entry.size} for entry in sorted(files,key=lambda x:x.path)}}
    fixed(OUT/'test_download_lock.json',data_lock)
    print('고정 Test 파일 수:',len(files),flush=True)
    print('원본 전체 용량 GiB:',round(sum(entry.size for entry in files)/1024**3,2),flush=True)
    print('정답을 사용하지 않고 문항 ID로 표본을 고릅니다.',flush=True)
    DATA.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id='MMMU/MMMU',repo_type='dataset',revision=REVISION,local_dir=DATA,
                      allow_patterns=[entry.path for entry in files],max_workers=2)
    grouped=defaultdict(list)
    total=0
    ids=set()
    for entry in sorted(files,key=lambda item:item.path):
        path=DATA/entry.path
        assert path.stat().st_size==entry.size
        values=pq.read_table(path,columns=['id']).column('id').to_pylist()
        subject=entry.path.split('/')[0]
        for index,candidate in enumerate(values):
            assert candidate not in ids
            ids.add(candidate);total+=1
            if candidate in seen:continue
            digest=hashlib.sha256(('independent-900-v1'+candidate).encode('utf-8')).hexdigest()
            grouped[subject].append({'id':candidate,'subject':subject,'file':entry.path,'row_index':index,'selection_sha256':digest})
    assert total==10500
    sample=[]
    for subject in sorted(expected_subjects):
        ordered=sorted(grouped[subject],key=lambda item:(item['selection_sha256'],item['id']))
        assert len(ordered)>=30
        sample.extend(ordered[:30])
    assert len(sample)==900 and not {row['id'] for row in sample}&seen
    sample_manifest={'repository':'MMMU/MMMU','revision':REVISION,'sample_size':900,'per_subject':30,
                     'selection':'SHA256 of independent-900-v1 plus ID, sorted within each subject. Exclude previously evaluated IDs. No gold labels, predictions or correctness used.',
                     'excluded_previously_seen_test_ids':len(ids&seen),'sample':sample}
    fixed(OUT/'sample_manifest.json',sample_manifest)
    answer_lock=OUT/'official_answer_lock.json'
    if answer_lock.exists():
        lock=read(answer_lock)
    else:
        commits=json.loads(fetch('https://api.github.com/repos/MMMU-Benchmark/MMMU/commits?path=mmmu/answer_dict_test.json&per_page=1'))
        assert isinstance(commits,list) and commits
        commit=commits[0]['sha']
        assert re.fullmatch(r'[0-9a-f]{40}',commit)
        lock={'repository':'MMMU-Benchmark/MMMU','commit':commit,'path':'mmmu/answer_dict_test.json'}
        fixed(answer_lock,lock)
    answer_path=OUT/'official_answer_dict_test.json'
    if not answer_path.exists():
        payload=fetch('https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/'+lock['commit']+'/'+lock['path'])
        json.loads(payload)
        temp=answer_path.with_suffix('.json.partial');temp.write_bytes(payload);temp.replace(answer_path)
    answers=read(answer_path)
    assert set(ids)<=set(answers)
    grading={}
    for row in sample:
        item=answers[row['id']]
        assert isinstance(item,dict) and 'ground_truth' in item and 'question_type' in item
        grading[row['id']]={'question_type':item['question_type'],'ground_truth':item['ground_truth']}
    fixed(OUT/'grading_reference.json',grading)
    records={entry.path:{'bytes':(DATA/entry.path).stat().st_size,'sha256':sha(DATA/entry.path)} for entry in files}
    report={'상태':'독립 900문항 원본 및 표본 준비 완료',
            '고정 문항 수':len(sample),'과목 수':30,'과목당 문항 수':30,
            '기존 평가와 겹쳐 제외한 Test ID 수':len(ids&seen),
            '표본 manifest SHA256':sha(OUT/'sample_manifest.json'),'정답 파일 SHA256':sha(answer_path),
            '원본 파일':records,'모델 추론':False,
            '다음 단계':'입력과 학습 데이터 중복 점검 및 정답을 읽지 않는 고정 모델 추론. 현재 스크립트는 다운로드와 표본 선정만 수행.',
            '주의':'전체 10500문항 점수와 구분합니다. 기존 평가 문항 ID는 제외했지만 모든 이미지 변형과 학습 데이터 중복을 배제한 것은 아닙니다.'}
    save(OUT/'preparation_summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='원본 파일'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
