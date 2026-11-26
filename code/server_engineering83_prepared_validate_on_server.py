"""원본 연결, 이미지 대조 및 실제 학습 전처리의 토큰·정답 마스크 검사."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def require(condition,message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_package(base):
    for relative,expected in read_json(base/'checksums.json').items():
        path=(base/relative).resolve()
        require(path.is_relative_to(base.resolve()),'검증 경로가 준비 폴더 밖을 가리킵니다.')
        require(path.is_file(),f'파일이 없습니다: {relative}')
        require(sha(path)==expected,f'파일 SHA256이 다릅니다: {relative}')

def compare_review_image(original,reference):
    """원본과 검토 이미지의 동일성을 허용된 축소 방식으로 정확히 대조합니다."""
    from PIL import Image
    original=original.convert('RGB')
    reference=reference.convert('RGB')
    if original.size==reference.size and original.tobytes()==reference.tobytes():
        return '원본 RGB 픽셀 일치'
    for method in [Image.Resampling.LANCZOS,Image.Resampling.BICUBIC,Image.Resampling.BILINEAR,Image.Resampling.NEAREST]:
        thumb=original.copy()
        thumb.thumbnail((1800,1800),resample=method)
        if thumb.size==reference.size and thumb.tobytes()==reference.tobytes():
            return '1800픽셀 축소 일치: '+method.name
        resized=original.resize(reference.size,resample=method)
        if resized.tobytes()==reference.tobytes():
            return '검토 크기 재조정 일치: '+method.name
    raise RuntimeError('원본 이미지와 검토 이미지의 RGB 대조가 일치하지 않습니다. 자동으로 통과시키지 않았습니다.')

def extract_originals(rows,source_root,base,destination):
    import pyarrow.parquet as pq
    from PIL import Image
    by_file=defaultdict(list)
    for item in rows:
        by_file[item['parquet_file']].append(item)
    linked=[]
    all_offsets=set()
    for name,items in by_file.items():
        path=(source_root/name).resolve()
        require(path.is_relative_to(source_root.resolve()),'원본 경로가 지정 폴더 밖을 가리킵니다.')
        require(path.is_file(),f'원본 Parquet가 없습니다: {path}')
        parquet=pq.ParquetFile(path)
        metadata=parquet.read(columns=['id','source','question','answer']).to_pylist()
        wanted={}
        for item in items:
            matches=[i for i,row in enumerate(metadata) if
                str(row['id'])==str(item['source_id']) and row['source']==item['source'] and
                row['question'].strip()==item['original_question'].strip() and
                row['answer'].strip()==item['original_answer'].strip()]
            require(len(matches)==1,f"원본 ID·질문·정답 연결이 유일하지 않습니다: {item['id']} ({len(matches)}개)")
            index=matches[0]
            offset=index-int(item['recorded_row'])
            require(offset in [0,-1],f"기록된 행 번호와 실제 위치가 다릅니다: {item['id']}")
            all_offsets.add(offset)
            require(index not in wanted,'동일 원본 행이 여러 후보에 연결됐습니다.')
            wanted[index]=item
        absolute_index=0
        found=0
        for batch in parquet.iter_batches(batch_size=32,columns=['image']):
            for j,value in enumerate(batch.column(0).to_pylist()):
                index=absolute_index+j
                if index not in wanted:
                    continue
                item=wanted[index]
                require(isinstance(value,dict) and value.get('bytes'),f"원본 이미지 바이트가 없습니다: {item['id']}")
                with Image.open(BytesIO(value['bytes'])) as loaded:
                    loaded.load()
                    require(list(loaded.size)==item['original_size'],f"원본 이미지 크기가 다릅니다: {item['id']}")
                    rgb=loaded.convert('RGB')
                with Image.open(base/'review_images'/f"{item['id']}.png") as reference:
                    match=compare_review_image(rgb,reference)
                image_path=destination/'images'/f"{item['id']}.png"
                rgb.save(image_path)
                rgb_hash=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                linked.append({'id':item['id'],'source':item['source'],'source_id':item['source_id'],
                    'parquet_file':name,'resolved_zero_based_row':index,'recorded_row_offset':offset,
                    'image_rgb_sha256':rgb_hash,'image_file_sha256':sha(image_path),
                    'image_match':match,'size':list(rgb.size),
                    'raw_question_sha256':hashlib.sha256(metadata[index]['question'].encode('utf-8')).hexdigest(),
                    'comparison_normalization':'질문과 정답의 양 끝 공백만 제거해 대조했습니다.'})
                found+=1
            absolute_index+=batch.num_rows
        require(found==len(items),f'이미지 추출 수가 일치하지 않습니다: {name}')
        print(f'원본 연결·이미지 대조: {name} / {found}개 통과',flush=True)
    require(len(all_offsets)==1,'원본 행 번호의 기준이 문항마다 다릅니다.')
    require(len(linked)==len(rows),'전체 원본 연결 수가 일치하지 않습니다.')
    return linked

def check_tokens(base,destination,rows,model,repo,settings):
    require(repo.is_dir(),f'학습 저장소가 없습니다: {repo}')
    sys.path.insert(0,str(repo))
    import torch
    from transformers import AutoProcessor
    from qwenvl.train.argument import DataArguments
    from qwenvl.data import data_processor as dp
    require(hasattr(dp,'preprocess_qwen_visual') and hasattr(dp,'update_processor_pixels'),
            '학습 전처리 함수가 예상과 다릅니다. 코드 내용을 확인해야 합니다.')
    processor=AutoProcessor.from_pretrained(str(model),local_files_only=True)
    args=DataArguments()
    args.min_pixels=settings['min_pixels']
    args.max_pixels=settings['max_pixels']
    processor=dp.update_processor_pixels(processor,args)
    tokenizer=processor.tokenizer
    image_token_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
    eos_id=tokenizer.convert_tokens_to_ids('<|im_end|>')
    by_kind={kind:read_json(base/f'candidate_{kind}.json') for kind in ['answer_only','concise_reasoning']}
    require(all(len(values)==len(rows) for values in by_kind.values()),'지도 형식별 문항 수가 다릅니다.')
    results=[]
    for index,item in enumerate(rows):
        targets={}
        for kind,values in by_kind.items():
            record=values[index]
            require(record['id']==item['id'],'문항 순서가 다릅니다.')
            require(record['conversations'][0]==by_kind['answer_only'][index]['conversations'][0],
                    '두 지도 형식의 질문이 다릅니다.')
            sample={**record,'data_path':str(destination)}
            # 데이터 로더의 오류 시 다른 문항으로 대체하는 경로를 거치지 않습니다.
            processed=dp.preprocess_qwen_visual([sample],processor)
            ids=processed['input_ids'].reshape(-1)
            labels=processed['labels'].reshape(-1)
            require(ids.shape==labels.shape,'입력과 정답 마스크 길이가 다릅니다.')
            supervised=labels[labels!=-100]
            require(supervised.numel()>0,'학습할 정답 토큰이 없습니다.')
            require(len(ids)<=settings['model_max_length'],f"전체 토큰 한도 초과: {item['id']} / {kind}")
            require(len(supervised)<=settings['max_target_tokens'],f"정답 토큰 한도 초과: {item['id']} / {kind}")
            require(bool((ids==image_token_id).any()),'이미지 토큰이 없습니다.')
            require(bool(torch.all(labels[ids==image_token_id]==-100)),'이미지 토큰이 정답으로 학습됩니다.')
            require('pixel_values' in processed and processed['pixel_values'].numel()>0,'이미지 입력이 비었습니다.')
            require('image_grid_thw' in processed and processed['image_grid_thw'].numel()==3,'이미지 그리드가 예상과 다릅니다.')
            require(bool((supervised==eos_id).any()),'종료 토큰이 정답 마스크에서 빠졌습니다.')
            decoded=tokenizer.decode(supervised.tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assistant=record['conversations'][1]['value']
            expected=assistant+'<|im_end|>'
            require(decoded.rstrip('\n')==expected,f"정답 마스크에 질문이 섞였거나 정답이 잘렸습니다: {item['id']} / {kind}")
            require(assistant.splitlines()[-1]=='Answer: '+item['final_answer'],'마지막 답이 짧은 풀이와 다릅니다.')
            targets[kind]={'total_tokens':len(ids),'target_tokens':len(supervised),
                'image_tokens':int((ids==image_token_id).sum()),'image_grid_thw':processed['image_grid_thw'].tolist()}
        results.append({'id':item['id'],**targets,'passed':True})
        if (index+1)%10==0 or index+1==len(rows):
            print(f'실제 학습 전처리·정답 마스크 검사: {index+1}/{len(rows)}개 통과',flush=True)
    versions={name:importlib.metadata.version(name) for name in ['torch','transformers','Pillow','pyarrow']}
    return results,versions

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,default=Path('/shared_dir/mmdl/data/external_candidates/mmfinereason_123k/cda6140bfba449bef9e57065a528b7424966e144'))
    parser.add_argument('--model',type=Path,default=Path('/shared_dir/mmdl/models/Qwen3-VL-4B-Instruct'))
    parser.add_argument('--repo',type=Path,default=Path('/shared_dir/mmdl/code/Qwen3-VL/qwen-vl-finetune'))
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    verify_package(base)
    rows=read_json(base/'paired_candidates.json')
    settings=read_json(base/'manifest.json')
    require(len(rows)==settings['paired_candidates'],'문항 수가 기록과 다릅니다.')
    destination=base/('validation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (destination/'images').mkdir(parents=True,exist_ok=False)
    report={'status':'검사 시작','training_authorized':False}
    try:
        linked=extract_originals(rows,args.source_root,base,destination)
        write_json(destination/'source_links.json',linked)
        tokens,versions=check_tokens(base,destination,rows,args.model,args.repo,settings)
        for kind in ['answer_only','concise_reasoning']:
            shutil.copyfile(base/f'candidate_{kind}.json',destination/f'candidate_{kind}.json')
        image_counts=Counter(x['image_rgb_sha256'] for x in linked)
        report.update(status='원본 연결·이미지·학습 전처리 검사 통과',question_count=len(rows),
            token_checks=tokens,versions=versions,original_image_groups=len(image_counts),
            additional_same_image_questions=sum(n-1 for n in image_counts.values()),
            max_total_tokens=max(x[k]['total_tokens'] for x in tokens for k in ['answer_only','concise_reasoning']),
            max_target_tokens=max(x[k]['target_tokens'] for x in tokens for k in ['answer_only','concise_reasoning']),
            preprocessing_file_sha256=sha(Path(inspect.getfile(__import__('qwenvl.data.data_processor',fromlist=['x'])))),
            pending=settings['pending'][2:])
    except Exception as error:
        report.update(status='검사 중단',error_type=type(error).__name__,error=str(error))
        write_json(destination/'validation_report.json',report)
        print('검사를 중단했습니다:',error,flush=True)
        print('기록:',destination/'validation_report.json',flush=True)
        return 1
    write_json(destination/'validation_report.json',report)
    print('\n=== 원본 연결·이미지·토큰 검사 완료 ===')
    print('문항 수:',report['question_count'])
    print('최대 전체 토큰 수:',report['max_total_tokens'])
    print('최대 정답 토큰 수:',report['max_target_tokens'])
    print('같은 이미지의 추가 문항 수:',report['additional_same_image_questions'])
    print('결과 폴더:',destination)
    print('후속 작업: 오염·근접 중복 검사와 원출처 확인 후 데이터 분할을 정합니다.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
