from pathlib import Path
from datetime import datetime
import copy
import fcntl
import hashlib
import importlib.util
import json
import traceback

ROOT = Path('/shared_dir/mmdl')
PREVIOUS = ROOT / 'checkpoints/verified_hard_compare_20261121'
JOB = ROOT / 'checkpoints/verified_hard_reasoning_20261121'
MODULE = None

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def status(state, **extra):
    save(JOB / 'status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

def paired_rows(rows, reasoning, selected):
    result = copy.deepcopy(rows)
    changed = []
    for index, row in enumerate(rows):
        if row.get('id') not in selected:
            continue
        pair = MODULE.absolute_row(reasoning[row['id']], MODULE.POOL)
        assert row['image'] == pair['image'], f'풀이형 이미지가 다릅니다: {row["id"]}'
        assert row['conversations'][0] == pair['conversations'][0], f'풀이형 질문이 다릅니다: {row["id"]}'
        assert len(row['conversations']) == len(pair['conversations']) == 2
        target = pair['conversations'][-1]['value'].strip()
        gold = row['conversations'][-1]['value'].strip()
        assert gold.startswith('Answer: ')
        assert target.endswith(gold), f'풀이형 최종 정답이 다릅니다: {row["id"]}'
        assert target != gold, f'풀이가 없는 문항입니다: {row["id"]}'
        assert pair['conversations'][-1]['from'] == 'gpt'
        result[index]['conversations'][-1] = copy.deepcopy(pair['conversations'][-1])
        assert result[index]['conversations'][0] == row['conversations'][0]
        changed.append(index)
    for index in set(range(len(rows))) - set(changed):
        assert result[index] == rows[index]
    return result, changed

def prepare():
    source = PREVIOUS / 'data/hard_answer'
    old_plan = read(PREVIOUS / 'experiment_plan.json')
    expected = next(row['학습 SHA256'] for row in old_plan['구성'] if row['구성'] == 'hard_answer')
    assert sha(source / 'train.json') == expected, '정답형 학습 데이터가 변경됐습니다.'
    rows = read(source / 'train.json')
    dev = read(source / 'internal_dev.json')
    selection = read(source / 'selection.json')
    selected = {item['id'] for item in selection}
    reasoning_file = MODULE.POOL / 'concise_reasoning_train.json'
    reasoning_rows = read(reasoning_file)
    reasoning = {row['id']: row for row in reasoning_rows}
    engineering_dev = read(MODULE.POOL / 'answer_only_internal_dev.json')
    assert len(reasoning_rows) == len(reasoning) == 63 and len(engineering_dev) == 16
    assert not selected & {row['id'] for row in engineering_dev}
    assert selected <= set(reasoning)
    assert len(rows) == 4000 and len(dev) == 400
    paired, changed = paired_rows(rows, reasoning, selected)
    repetitions = 4
    assert len(changed) == len(selected) * repetitions
    assert set(changed) == set(old_plan['공통 교체 위치'])
    assert sum(bool(row.get('image')) for row in paired) == 3200
    assert all(paired[i] == rows[i] for i in range(4000) if not rows[i].get('image'))
    folder = JOB / 'data/hard_reasoning'
    folder.mkdir(parents=True, exist_ok=True)
    MODULE.fixed_json(folder / 'train.json', paired)
    MODULE.fixed_json(folder / 'internal_dev.json', dev)
    MODULE.fixed_json(folder / 'selection.json', selection)
    plan = {'목적': '같은 어려운 공학 문항의 정답형과 풀이형 비교',
            '정답형 대조군': str(PREVIOUS / 'hard_answer'),
            '학습 문항 수': 4000, '이미지 문항 수': 3200, '텍스트 Replay 수': 800,
            '풀이 적용 고유 문항 수': len(selected), '풀이 적용 학습 기록 수': len(changed),
            '학습률': 0.0002, 'rank': 8, 'alpha': 16, '학습 단계': 500, '시드': 42,
            '고정 조건': '정답형과 같은 질문, 이미지, 문항 순서, 분할, 반복 수와 지시문. 선택된 응답만 풀이형으로 교체.',
            '시작 모델': '순정 모델에서 시작. 기존 어댑터에 추가 학습하지 않음.',
            '풀이 출처': str(reasoning_file), '풀이 원본 SHA256': sha(reasoning_file),
            '정답형 학습 SHA256': expected, '풀이형 학습 SHA256': sha(folder / 'train.json'),
            '개발 SHA256': sha(folder / 'internal_dev.json'), '변경 위치': changed,
            '유료 API': False, '새 teacher 추론': False,
            '한계': '단일 시드의 소규모 비교. 문항과 학습 단계는 같지만 풀이형의 지도 토큰 수는 더 많음. 공학 표기 일치를 의미적 정확도로 해석하지 않음.'}
    MODULE.fixed_json(JOB / 'experiment_plan.json', plan)
    return plan

def main():
    global MODULE
    path = ROOT / 'train_verified_hard_server.py'
    spec = importlib.util.spec_from_file_location('hard_comparison_helper', path)
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)
    MODULE.JOB = JOB
    MODULE.ARMS = ['hard_reasoning']
    MODULE.prepare = prepare
    MODULE.fixed_json(JOB / 'engineering_reference_R20.json', read(PREVIOUS / 'engineering_reference_R20.json'))
    MODULE.main()
    previous = read(PREVIOUS / 'hard_comparison.json')
    current = read(JOB / 'hard_comparison.json')
    added = next(row for row in current['결과'] if row['구성'] == 'hard_reasoning')
    base_before = next(row for row in previous['결과'] if row['구성'] == 'reference_R20')
    base_after = next(row for row in current['결과'] if row['구성'] == 'reference_R20')
    assert base_before == base_after, '기준 R20 결과가 이전 평가와 다릅니다.'
    report = [*previous['결과'], added]
    save(JOB / 'answer_reasoning_comparison.json', {'결과': report, '설계': read(JOB / 'experiment_plan.json'),
         '주의': '공학 정답 손실과 표기 일치는 보조 지표입니다. 생성 답변의 수식 동치와 정답성을 확인한 후 성능 향상을 판단해야 합니다. MMMU 평가는 아직 실행하지 않았습니다.'})
    status('정답형과 풀이형 개발 비교 완료', 결과=report, 결과파일=str(JOB / 'answer_reasoning_comparison.json'))
    print(json.dumps(read(JOB / 'status.json'), ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'checkpoints/rank_comparison.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('다른 학교 GPU 비교 작업이 실행 중입니다.')
        try:
            main()
        except Exception:
            status('오류로 중단', 오류=traceback.format_exc())
            raise
