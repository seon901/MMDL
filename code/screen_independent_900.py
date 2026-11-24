from pathlib import Path
from collections import defaultdict, Counter
from io import BytesIO
import ast
import hashlib
import json
import re
import struct
import unicodedata

PROJECT=Path(__file__).resolve().parent.parent
OUT=PROJECT/'evaluation_reference/independent_900_20261121'
RAW=PROJECT/'transfer_packages/mmdl_source_20261114_220144_576558'
TEST=PROJECT/'evaluation_reference/MMMU_Test/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
PRO=PROJECT/'evaluation_reference/MMMU_Pro/0d7426df4ccb3d8704a992fe6850abee7c262127'
A=PROJECT/'eval_candidates/recipe_a_answer_only_lr_2e-4_step500_20261116_232036_170898/records'
HARD=PROJECT/'eval_candidates/mmmu_hard_20261121_043545_392882/records/data/hard_answer/train.json'
ENGINEERING=PROJECT/'server_backups/engineering_update_20261119/engineering79_validation'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def fixed(path,value):
    if path.exists():assert read(path)==value,f'고정 기록과 다릅니다: {path}'
    else:save(path,value)
def norm_question(value):
    value=unicodedata.normalize('NFKC',value).casefold()
    value=re.sub(r'<image(?:\s+\d+)?>',' ',value)
    value=value.split('\noptions:')[0]
    value=re.sub(r'^\s*question\s*:\s*','',value)
    return ' '.join(value.split())
def grams(value):
    tokens=re.findall(r'\w+',value)
    return frozenset(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))

def main():
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    assert sha(OUT/'sample_manifest.json')=='772ae6e8b8530c6adde67734ec8b9fd41afe4cdac0046858ff1de18096cc880d'
    settings={'version':'input-screening-v1','image_exact':'RGB pixels with width and height',
              'image_near':'NumPy DCT 64-bit pHash, Hamming distance <=4. Near-hash checks disabled below grayscale standard deviation 2.',
              'text':'Normalized question exact match, or exact word 3-gram Jaccard >=0.6 when both sets have at least 10 grams.',
              'replacement':'Before any model predictions or scores. Within each subject, continue the frozen SHA256 ordering until 30 candidates pass. Preserve original sample manifest and record all rejections.',
              'labels':'No gold labels, correctness, explanations or model predictions are read by this screening script.',
              'limitations':'Conservative similarity screening may exclude unrelated examples. Does not establish absence of all paraphrases, transformations or pretraining exposure.',
              'code_sha256':sha(Path(__file__))}
    fixed(OUT/'screening_protocol.json',settings)
    checksums={}
    for line in (RAW/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        h,name=line.split(None,1);checksums[name.strip().lstrip('*')]=h
    questions=[];q_exact={};inverted=defaultdict(set)
    image_exact={};image_hashes=[]
    basis=np.cos(np.pi*np.arange(8)[:,None]*(2*np.arange(32)[None,:]+1)/64)
    def add_question(text,reference):
        value=norm_question(text)
        if not value:return
        if value in q_exact:return
        g=grams(value);index=len(questions)
        questions.append((value,g,reference));q_exact[value]=reference
        for token in g:inverted[token].add(index)
    def features(image):
        rgb=image.convert('RGB')
        exact=hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()
        gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        dct=(basis@gray@basis.T).reshape(-1)
        threshold=float(np.median(dct[1:]));bits=dct>threshold
        value=sum(int(bit)<<index for index,bit in enumerate(bits))
        return exact,value,float(gray.std())
    def add_image(image,reference):
        exact,value,contrast=features(image)
        if exact not in image_exact:
            image_exact[exact]=reference
            if contrast>=2:image_hashes.append((value,reference))
    def from_cell(cell):
        assert cell and cell.get('bytes'),'이미지 내장 데이터를 찾지 못했습니다.'
        return Image.open(BytesIO(cell['bytes']))
    desired=defaultdict(set)
    for split,filename in [('train','train_provenance.json'),('internal_dev','internal_dev_provenance.json')]:
        entries=read(A/filename)
        assert len(entries)==(4000 if split=='train' else 400)
        for entry in entries:
            tag=f"A_{split}:{entry['출처']}:{entry['원본 행 번호']}:{entry['QA 번호']}"
            add_question(entry['질문'],tag)
            desired[entry['원본 파일']].add(int(entry['원본 행 번호']))
    for filename,indices in desired.items():
        path=RAW/filename
        assert sha(path)==checksums[filename],f'원본 SHA256 불일치: {filename}'
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['image']):
            for local,row in enumerate(batch.to_pylist()):
                index=offset+local
                if index in indices:
                    with from_cell(row['image']) as image:add_image(image,f'A_source:{filename}:{index}')
            offset+=len(batch)
        print('학습 및 개발 이미지 검사 기준 준비:',filename,flush=True)
    texts=[row for row in read(HARD) if not row.get('image')]
    assert len(texts)==800
    for index,row in enumerate(texts):add_question(row['conversations'][0]['value'],f'replay_text:{index}')
    links={row['id']:row for row in read(ENGINEERING/'source_links.json')}
    for row in read(ENGINEERING/'candidate_answer_only.json'):
        reference='engineering_reviewed:'+row['id']
        add_question(row['conversations'][0]['value'],reference)
        path=ENGINEERING/row['image']
        assert sha(path)==links[row['id']]['image_file_sha256']
        with Image.open(path) as image:add_image(image,reference)
    seen=set()
    for path in sorted((PRO/'standard (10 options)').glob('*.parquet')):
        columns=['id','question']+[f'image_{n}' for n in range(1,8)]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=columns):
            for row in batch.to_pylist():
                seen.add(row['id']);reference='previous_Pro:'+row['id']
                add_question(row['question'],reference)
                for n in range(1,8):
                    if row.get(f'image_{n}'):
                        with from_cell(row[f'image_{n}']) as image:add_image(image,reference)
    assert len(seen)==1730
    print('검사 기준 고유 질문:',len(questions),'고유 이미지:',len(image_exact),flush=True)
    def screen_question(text):
        value=norm_question(text)
        if value in q_exact:return {'type':'question_exact','reference':q_exact[value]}
        g=grams(value)
        if len(g)<10:return None
        candidates=set()
        for token in g:candidates.update(inverted.get(token,()))
        for index in sorted(candidates):
            _,other,reference=questions[index]
            if len(other)<10 or min(len(g),len(other))/max(len(g),len(other))<0.6:continue
            common=len(g&other);score=common/(len(g)+len(other)-common)
            if score>=0.6:return {'type':'question_near','reference':reference,'jaccard':score}
        return None
    def screen_image(image):
        exact,value,contrast=features(image)
        if exact in image_exact:return {'type':'image_exact','reference':image_exact[exact]}
        if contrast>=2:
            for other,reference in image_hashes:
                distance=(value^other).bit_count()
                if distance<=4:return {'type':'image_near','reference':reference,'hamming':distance}
        return None
    preparation=read(OUT/'preparation_summary.json')
    files=preparation['원본 파일']
    grouped=defaultdict(list)
    for filename,record in sorted(files.items()):
        path=TEST/filename
        assert sha(path)==record['sha256'],f'Test 원본 SHA256 불일치: {filename}'
        ids=pq.read_table(path,columns=['id']).column('id').to_pylist()
        for index,candidate in enumerate(ids):
            if candidate in seen:continue
            grouped[filename.split('/')[0]].append({'id':candidate,'subject':filename.split('/')[0],
                   'file':filename,'row_index':index,'selection_sha256':hashlib.sha256(('independent-900-v1'+candidate).encode('utf-8')).hexdigest()})
    assert len(grouped)==30
    selected=[];rejected=[];input_rows=[]
    image_directory=OUT/'screened_images';image_directory.mkdir(exist_ok=True)
    for subject in sorted(grouped):
        choices=sorted(grouped[subject],key=lambda row:(row['selection_sha256'],row['id']))
        tables={name:pq.read_table(TEST/name,columns=['id','question','options','question_type']+[f'image_{i}' for i in range(1,8)]) for name in sorted({row['file'] for row in choices})}
        accepted=0
        for candidate in choices:
            row=tables[candidate['file']].slice(candidate['row_index'],1).to_pylist()[0]
            assert row['id']==candidate['id']
            flag=screen_question(row['question'])
            if not flag:
                for n in range(1,8):
                    cell=row.get(f'image_{n}')
                    if cell:
                        with from_cell(cell) as image:flag=screen_image(image)
                        if flag:break
            if flag:
                rejected.append({**candidate,'reason':flag});continue
            inputs={'id':row['id'],'subject':subject,'question':row['question'],'options':row['options'],
                    'question_type':row['question_type'],'images':{}}
            for n in range(1,8):
                cell=row.get(f'image_{n}')
                if cell:
                    path=image_directory/(row['id']+f'_image_{n}.png')
                    with from_cell(cell) as image:image.convert('RGB').save(path,format='PNG')
                    inputs['images'][f'image_{n}']=str(path.relative_to(PROJECT)).replace('\\','/')
            canonical=json.dumps(inputs,ensure_ascii=False,sort_keys=True).encode('utf-8')
            inputs['input_text_sha256']=hashlib.sha256(canonical).hexdigest()
            inputs['image_sha256']={key:sha(PROJECT/value) for key,value in inputs['images'].items()}
            input_rows.append(inputs);selected.append(candidate);accepted+=1
            if accepted==30:break
        assert accepted==30,f'{subject}: 검사를 통과한 문항이 부족합니다.'
        del tables
        print(subject,': 30문항 선정 완료',flush=True)
    assert len(selected)==len(input_rows)==900
    original=read(OUT/'sample_manifest.json')
    changed=set(row['id'] for row in original['sample'])-{row['id'] for row in selected}
    manifest={'repository':original['repository'],'revision':original['revision'],'sample_size':900,'per_subject':30,
              'screening_protocol_sha256':sha(OUT/'screening_protocol.json'), 'sample':selected,
              'initial_manifest_sha256':sha(OUT/'sample_manifest.json'),'replaced_initial_ids':sorted(changed),
              'selection':'Continue frozen per-subject hash order after label-blind input similarity screening. No model predictions or scores used.'}
    fixed(OUT/'screened_sample_manifest.json',manifest)
    fixed(OUT/'blind_inputs.json',input_rows)
    save(OUT/'screening_flags.json',rejected)
    save(OUT/'screening_summary.json',{'상태':'입력 중복 검사와 정답 없는 평가 입력 준비 완료',
         '문항 수':900,'과목 수':30,'검사 기준 고유 질문 수':len(questions),'검사 기준 고유 이미지 수':len(image_exact),
         '검사 중 보류한 후보 수':len(rejected),'처음 표본에서 교체한 문항 수':len(changed),
         '보류 사유':dict(Counter(row['reason']['type'] for row in rejected)),
         '최종 표본 manifest SHA256':sha(OUT/'screened_sample_manifest.json'),
         '정답 없는 입력 SHA256':sha(OUT/'blind_inputs.json'),'모델 추론':False,
         '한계':settings['limitations']})
    print(json.dumps(read(OUT/'screening_summary.json'),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
