from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import subprocess
import time

PROJECT = Path('/mnt/c/Users/K/Desktop/MMDL')
JOB = PROJECT / 'logs/A_reproduction_20261120'
RESULT = PROJECT / 'results/A_reproduction_20261120'

def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None

def check():
    state = read(JOB / 'status.json') or {}
    comparison = read(JOB / 'reproduction_comparison.json')
    summary = read(RESULT / '_summary.json')
    subjects = list((read(PROJECT / 'results_vllm_v2/_summary.json') or {}).get('subject_accuracy', {}))
    count = 0
    for subject in subjects:
        rows = read(RESULT / (subject + '.json'))
        count += int(isinstance(rows, list) and len(rows) == 30)
    processes = subprocess.check_output(['ps', '-eo', 'pid,etime,stat,pcpu,args'], text=True)
    running = []
    for line in processes.splitlines()[1:]:
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        command = fields[4]
        if str(JOB / 'run.py') in command or ('eval_mmmu_vllm.py' in command and str(RESULT) in command):
            running.append(fields)
    complete = count == 30 and summary is not None and comparison is not None and state.get('상태') == '기준 A 재현 평가 완료'
    print('확인 시각:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if complete:
        print('전체 상태: 완료')
    elif state.get('상태') == '오류로 중단':
        print('전체 상태: 오류로 중단')
    elif running:
        print('전체 상태: 실행 중')
    else:
        print('전체 상태: 실행 프로세스 없음. 미완료라면 중단 또는 시작 실패')
    print('현재 단계:', state.get('상태', '아직 상태 기록 없음'))
    print(f'완료 과목: {count}/30')
    print(f'저장 완료 문항: {count*30}/900')
    if complete:
        print('\n=== 최종 재현 비교 ===')
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print('결과 파일:', JOB / 'reproduction_comparison.json')
    else:
        if running:
            print('\n실행 프로세스:')
            for pid, elapsed, condition, cpu, _ in running:
                print(f'프로세스 {pid} | 실행 시간 {elapsed} | CPU {cpu}% | 상태 {condition}')
        log = Path(state.get('로그') or str(JOB / 'runner.log'))
        print('\n최근 로그:', log)
        if log.is_file():
            print('마지막 로그 갱신:', max(0, int(time.time()-log.stat().st_mtime)), '초 전')
            lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
            print('\n'.join(lines[-10:]) if lines else '아직 로그 내용이 없습니다.')
        else:
            print('아직 로그 파일이 없습니다.')
        if state.get('오류'):
            print('\n오류 기록:\n' + state['오류'])
    return complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    arguments = parser.parse_args()
    assert PROJECT.is_dir(), '개인 컴퓨터 WSL에서 실행해 주세요.'
    try:
        while True:
            if arguments.watch and os.environ.get('TERM'):
                print('\033[2J\033[H', end='')
            complete = check()
            if complete or not arguments.watch:
                break
            print('\n10초마다 갱신합니다. Ctrl+C는 확인 화면만 종료합니다.')
            time.sleep(10)
    except KeyboardInterrupt:
        print('\n확인 화면을 종료했습니다. 백그라운드 평가는 계속 실행됩니다.')