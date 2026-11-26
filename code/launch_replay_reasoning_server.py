from pathlib import Path
import ast
import fcntl
import hashlib
import json
import subprocess
import sys

ROOT = Path('/shared_dir/mmdl')
DATA = ROOT / 'data/replay_reasoning_pair_20261120_212449_530854'
JOB = ROOT / 'checkpoints/replay_reasoning_compare_20261120_212449_530854'
TRAIN_SOURCE = ROOT / 'checkpoints/ratio_compare_20261119_201447_595548/run.py'
TEXT_SOURCE = ROOT / 'checkpoints/replay_compare_20261119_214637_363626/evaluate_text.py'
OLD_FORMAT = ROOT / 'checkpoints/native_reasoning_compare_20261119_232833_892795'
LABELS = ['replay_answer_control', 'replay_native_reasoning']
HASHES = {'replay_answer_control': '33c9691dbbe6669396dd5a0b4e7d928ba16154d3fc0f2bb95bda8b9e447700be', 'replay_native_reasoning': '58ddfe4f9ba742eb676c4fcd18ecd40ceea4b6bdc705874f0ec921a16ec31e33'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fixed(path, content):
    if path.exists():
        assert path.read_bytes() == content, f'내용이 다른 파일이 있습니다: {path}'
    else:
        path.write_bytes(content)

def script(path, tree):
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + '\n'
    compile(code, str(path), 'exec')
    assert all(ord(c)<128 or 0xAC00<=ord(c)<=0xD7A3 or ord(c)==183 for c in code), '코드 문자 검사 실패'
    fixed(path, code.encode('utf-8'))

print('=== 조합 실험 실행 준비 ===', flush=True)
lock = (ROOT / 'checkpoints/rank_comparison.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('다른 비교 작업이 실행 중입니다. 중복 실행하지 않습니다.')
plan = read(DATA / 'preparation_summary.json')
assert plan['풀이형 적용 문항 수'] == 722
for name, expected in HASHES.items():
    assert sha(DATA / name / 'train.json') == expected
    assert sha(DATA / name / 'internal_dev.json') == plan['개발 데이터 SHA256']
    assert sha(DATA / name / 'internal_dev_provenance.json') == plan['개발 원본 기록 SHA256']
assert TRAIN_SOURCE.is_file() and TEXT_SOURCE.is_file()
JOB.mkdir(exist_ok=True)
fixed(JOB / 'combo_plan.json', (DATA / 'preparation_summary.json').read_bytes())
tree = ast.parse(TRAIN_SOURCE.read_text(encoding='utf-8'))
changed = set()
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    names = {target.id for target in node.targets if isinstance(target, ast.Name)}
    if 'DATA' in names:
        node.value = ast.parse(f'Path({str(DATA)!r})', mode='eval').body
        changed.add('DATA')
    if 'HASHES' in names:
        node.value = ast.parse(repr(HASHES), mode='eval').body
        changed.add('HASHES')
assert changed == {'DATA', 'HASHES'}
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('latest_ratio_comparison.txt', 'latest_replay_reasoning_comparison.txt').replace('ratio_comparison.json', 'image_comparison.json')
        if node.value == '세 구성의 학습과 평가 완료':
            node.value = '조합 두 구성의 학습과 이미지 평가 완료'
script(JOB / 'train_runner.py', tree)

adapters = {'answer_control': str(OLD_FORMAT / 'answer_control'), 'native_reasoning': str(OLD_FORMAT / 'native_reasoning'), **{name: str(JOB / name) for name in LABELS}}
for path in adapters.values():
    if str(OLD_FORMAT) in path:
        assert read(Path(path) / 'trainer_state.json')['global_step'] == 500

class TextPatch(ast.NodeTransformer):
    def __init__(self):
        self.loops = 0
        self.folders = 0
    def visit_For(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id == 'label' and isinstance(node.iter, ast.List) and [item.value if isinstance(item, ast.Constant) else None for item in node.iter.elts] == ['R10', 'R20']:
            node.iter = ast.parse(repr(list(adapters)), mode='eval').body
            self.loops += 1
        return node
    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == 'folder' for target in node.targets) and isinstance(node.value, ast.BinOp) and isinstance(node.value.left, ast.Name) and node.value.left.id == 'JOB' and isinstance(node.value.right, ast.Name) and node.value.right.id == 'label':
            node.value = ast.parse('Path(ADAPTERS[label])', mode='eval').body
            self.folders += 1
        return node

tree = ast.parse(TEXT_SOURCE.read_text(encoding='utf-8'))
changed = 0
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'JOB' for target in node.targets):
        node.value = ast.parse(f'Path({str(JOB)!r})', mode='eval').body
        changed += 1
assert changed == 1
patch = TextPatch()
tree = patch.visit(tree)
assert patch.loops == patch.folders == 1, '기존 텍스트 평가 코드 구조가 다릅니다.'
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('ratio_comparison.json', 'image_comparison.json').replace('replay_plan.json', 'combo_plan.json').replace('replay_comparison.json', 'combo_comparison.json')
        if node.value == 'Replay 이미지와 텍스트 평가 완료':
            node.value = '조합 이미지와 텍스트 평가 완료'
launchers = [i for i, node in enumerate(tree.body) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__']
assert len(launchers) == 1
tree.body.insert(launchers[0], ast.parse('ADAPTERS = ' + repr(adapters)).body[0])
script(JOB / 'evaluate_text.py', tree)

orchestrator = r'''
from pathlib import Path
from datetime import datetime
import fcntl
import json
import subprocess
import sys
import traceback

JOB = Path(__file__).resolve().parent
OLD = Path('/shared_dir/mmdl/checkpoints/native_reasoning_compare_20261119_232833_892795')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def status(state, **extra):
    save(JOB / 'overall_status.json', {'상태': state, '갱신 시각': datetime.now().isoformat(), **extra})

with (JOB / 'orchestrator.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('같은 조합 실험이 실행 중입니다.')
    try:
        for phase, filename, logfile in [('학습과 이미지 평가', 'train_runner.py', 'training_stage.log'), ('텍스트 평가', 'evaluate_text.py', 'text_stage.log')]:
            status('조합 실험 실행 중', 단계=phase, 로그=str(JOB / logfile))
            with (JOB / logfile).open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', str(JOB / filename)], stdout=log, stderr=subprocess.STDOUT, check=True)
        combined = read(JOB / 'combo_comparison.json')
        old_image = {item['구성']: item['정답 수'] for item in read(OLD / 'format_comparison.json')['결과']}
        new_image = {item['구성']: item['정답 수'] for item in combined['이미지 개발 400개']}
        text = {item['구성']: item['정답 수'] for item in combined['텍스트 개발 300개']}
        cells = []
        for replay, form, label in [(0, '정답형', 'answer_control'), (0, '풀이형', 'native_reasoning'), (20, '정답형', 'replay_answer_control'), (20, '풀이형', 'replay_native_reasoning')]:
            cells.append({'Replay 비율': replay, '지도 형식': form, '모델': label, '이미지 정답 수': old_image[label] if replay == 0 else new_image[label], '텍스트 정답 수': text[label]})
        image_interaction = (new_image['replay_native_reasoning'] - new_image['replay_answer_control']) - (old_image['native_reasoning'] - old_image['answer_control'])
        text_interaction = (text['replay_native_reasoning'] - text['replay_answer_control']) - (text['native_reasoning'] - text['answer_control'])
        report = {'네 조건 결과': cells, '이미지 정답 수 기준 상호작용 관측차': image_interaction, '텍스트 정답 수 기준 상호작용 관측차': text_interaction, '주의': '단일 시드의 내부 개발 결과입니다. 지도 토큰 수는 형식별로 다릅니다. 최종 MMMU 우월성은 별도 고정 평가가 필요합니다.'}
        save(JOB / 'factorial_comparison.json', report)
        status('조합 학습과 이미지 및 텍스트 평가 완료', 결과=report, 결과파일=str(JOB / 'factorial_comparison.json'), 상세파일=str(JOB / 'combo_comparison.json'))
        print(json.dumps(read(JOB / 'overall_status.json'), ensure_ascii=False, indent=2), flush=True)
    except Exception:
        status('오류로 중단', 오류=traceback.format_exc(), 단계=phase, 로그=str(JOB / logfile))
        raise
'''
compile(orchestrator, str(JOB / 'orchestrate.py'), 'exec')
assert all(ord(c)<128 or 0xAC00<=ord(c)<=0xD7A3 or ord(c)==183 for c in orchestrator)
fixed(JOB / 'orchestrate.py', orchestrator.encode('utf-8'))
record = {'데이터': str(DATA), '학습률': 0.0002, 'rank': 8, 'alpha': 16, '단계': 500, '시드': 42, '풀이 대상': 722, '학습 실행 원본 SHA256': sha(TRAIN_SOURCE), '텍스트 평가 원본 SHA256': sha(TEXT_SOURCE), '텍스트 평가 대상': list(adapters)}
fixed(JOB / 'execution_plan.json', json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'))
(ROOT / 'checkpoints/latest_replay_reasoning_comparison.txt').write_text(str(JOB), encoding='utf-8')
fcntl.flock(lock, fcntl.LOCK_UN)
lock.close()
with (JOB / 'runner.log').open('a', encoding='utf-8') as log:
    log.write('\n=== Replay와 풀이 조합 실험 실행 요청 ===\n')
    log.flush()
    process = subprocess.Popen([sys.executable, '-u', str(JOB / 'orchestrate.py')], cwd=JOB, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print('=== 조합 실험 실행 요청 완료 ===', flush=True)
print('프로세스 번호:', process.pid, flush=True)
print('작업 폴더:', JOB, flush=True)