from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import subprocess
import time
PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
IDENTIFIER = 'mmmu_combo_20261120_222910_415889'
JOB = PROJECT / 'logs' / IDENTIFIER
LABELS = [('replay_native_reasoning', 'Replay와 짧은 풀이 조합'), ('replay_answer_control', 'Replay 정답형 대조군')]

def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None

def check():
    state = read(JOB / 'status.json') or {}
    subjects = list((read(PROJECT / 'results_vllm_v2/_summary.json') or {}).get('subject_accuracy', {}))
    processes = subprocess.check_output(['ps', '-eo', 'pid,etime,stat,pcpu,args'], text=True)
    main = []
    children = []
    for line in processes.splitlines()[1:]:
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        pid, elapsed, condition, cpu, command = fields
        if str(JOB / 'run.py') in command:
            (children if '--merge' in command else main).append((pid, elapsed, condition, cpu))
        elif 'eval_mmmu_vllm.py' in command and str(PROJECT / 'merged_models' / IDENTIFIER) in command:
            children.append((pid, elapsed, condition, cpu))
    details = []
    all_done = len(subjects) == 30
    for label, title in LABELS:
        folder = PROJECT / 'results' / IDENTIFIER / label
        count = 0
        for subject in subjects:
            rows = read(folder / (subject + '.json'))
            if isinstance(rows, list) and len(rows) == 30:
                count += 1
        summary = read(folder / '_summary.json')
        done = count == 30 and summary is not None
        all_done = all_done and done
        merged = (PROJECT / 'merged_models' / IDENTIFIER / label / 'merge_record.json').is_file()
        details.append((title, count, done, merged))
    comparison = read(JOB / 'mmmu_comparison.json')
    completed = all_done and comparison is not None and all((label in comparison for label, _ in LABELS))
    print('확인 시각:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if completed:
        print('전체 상태: 완료')
    elif state.get('상태') == '오류로 중단':
        print('전체 상태: 오류로 중단')
    elif main:
        paused = all((item[2].startswith(('T', 't')) for item in main + children))
        print('전체 상태:', '일시 정지' if paused else '실행 중')
    elif children:
        print('전체 상태: 하위 작업 실행 중. 순차 실행 담당 프로세스 확인 필요')
    else:
        print('전체 상태: 실행 프로세스 없음. 완료되지 않았다면 중단 또는 시작 실패')
    print('현재 단계:', state.get('상태', '아직 상태 기록 없음'))
    if state.get('모델'):
        print('현재 모델:', state['모델'])
    print()
    for title, count, done, merged in details:
        phase = '평가 완료' if done else '병합 완료' if merged else '병합 대기 또는 진행 중'
        print(f'{title}: {phase}, 평가 {count}/30과목, 저장된 문항 {count * 30}/900개')
    if comparison:
        print('\n현재까지 완료된 점수:')
        for label in ['base', 'A', 'replay_native_reasoning', 'replay_answer_control', 'R20']:
            if label in comparison:
                print(label, str(comparison[label]['정답 수']) + '/900')
    if main or children:
        print('\n실행 프로세스:')
        for pid, elapsed, condition, cpu in main + children:
            print(f'프로세스 {pid} | 실행 시간 {elapsed} | CPU {cpu}% | 상태 {condition}')
    log = Path(state.get('로그') or str(JOB / 'runner.log'))
    print('\n최근 로그:', log)
    if log.is_file():
        age = max(0, int(time.time() - log.stat().st_mtime))
        print(f'마지막 로그 갱신: {age}초 전')
        lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        print('\n'.join(lines[-8:]) if lines else '아직 로그 내용이 없습니다. 초기 준비 중일 수 있습니다.')
    else:
        print('아직 로그 파일이 없습니다.')
    if state.get('오류'):
        print('\n오류 기록:\n' + state['오류'])
    return completed
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    arguments = parser.parse_args()
    assert PROJECT.is_dir(), '개인 컴퓨터 WSL에서 실행해 주세요.'
    try:
        while True:
            if arguments.watch and os.environ.get('TERM'):
                print('\x1b[2J\x1b[H', end='')
            completed = check()
            if not arguments.watch or completed:
                break
            print('\n10초마다 갱신합니다. Ctrl+C는 확인 화면만 종료합니다.')
            time.sleep(10)
    except KeyboardInterrupt:
        print('\n확인 화면을 종료했습니다. 백그라운드 평가는 계속 실행됩니다.')
