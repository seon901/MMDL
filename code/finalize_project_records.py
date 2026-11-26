from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / 'results/final_validation_900_20261121'
EVALUATOR = ROOT / 'scripts/eval_mmmu_vllm.py'
EVAL_SHA = '0089184bc12014333652abf69abb766fceeb1f6a11e4f1792cb7826bcb2479b8'
RULE_SHA = 'd46c67918648558e038038f976777df69cdcd46e0c850d65f1ba449fd606cd22'
NAMES = ('base', 'A', 'R20', 'hard_answer', 'hard_reasoning')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_final():
    assert sha(EVALUATOR) == EVAL_SHA, '기존 평가 코드가 달라졌습니다.'
    rule_path = ROOT / 'scripts/frozen_ensemble_rule.py'
    assert sha(rule_path) == RULE_SHA, '고정 투표 코드가 달라졌습니다.'
    spec = importlib.util.spec_from_file_location('final_fixed_vote', rule_path)
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    wanted = {'normalize_str', 'grade_open'}
    nodes = [node for node in ast.parse(EVALUATOR.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    grading = {'re': re, 'ast': ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), 'exec'), grading)
    reference = read(FINAL / 'comparison.json')
    subjects = sorted(read(FINAL / 'base/_summary.json')['subject_accuracy'])
    assert len(subjects) == 30
    predictions = {}
    report = {}
    for name in NAMES:
        values = {}
        summary = read(FINAL / name / '_summary.json')
        assert set(summary['subject_accuracy']) == set(subjects)
        for key, expected in {'max_new_tokens': 512, 'min_pixels': 200704, 'max_pixels': 802816, 'seed': 3407, 'presence_penalty': 1.5, 'backend': 'vllm'}.items():
            assert summary[key] == expected, (name, key)
        for subject in subjects:
            rows = read(FINAL / name / (subject + '.json'))
            assert len(rows) == 30
            for row in rows:
                assert row['subject'] == subject and row['id'] not in values
                expected = row['predicted'] == row['answer'] if row['is_mcq'] else grading['grade_open'](row['answer'], row['predicted'])
                assert bool(row['correct']) == bool(expected), (name, row['id'], '채점 불일치')
                values[row['id']] = row
        assert len(values) == 900
        predictions[name] = values
        correct = sum(row['correct'] for row in values.values())
        assert correct == reference['results'][name]['correct']
        assert abs(summary['overall_macro_avg'] - correct / 900) < 1e-10
        report[name] = {'correct': correct, 'total': 900, 'accuracy': correct / 900}
    ids = sorted(predictions['base'])
    for name, values in predictions.items():
        assert set(values) == set(ids)
        assert all(all(values[key][field] == predictions['base'][key][field] for field in ('subject', 'answer', 'is_mcq')) for key in ids)
    for group, names in rule.GROUPS.items():
        stored_votes = {row['id']: row for row in read(FINAL / 'votes' / (group + '.json'))}
        stored_scores = {row['id']: row for row in read(FINAL / 'scored' / (group + '.json'))}
        assert set(stored_votes) == set(stored_scores) == set(ids)
        total = 0
        for key in ids:
            answer, selected = rule.choose_answer({name: predictions[name][key]['predicted'] for name in names}, group)
            assert stored_votes[key]['predicted'] == answer and stored_votes[key]['selected_model'] == selected
            assert stored_scores[key]['predicted'] == answer
            assert stored_scores[key]['correct'] == predictions[selected][key]['correct']
            total += int(predictions[selected][key]['correct'])
        assert total == reference['results'][group]['correct']
        report[group] = {'correct': total, 'total': 900, 'accuracy': total / 900}
    return {'status': 'passed', 'same_validation_ids': 900, 'same_gold_answers': True,
            'same_grading_recomputed': True, 'frozen_votes_recomputed': True,
            'results': report, 'evaluator_sha256': EVAL_SHA, 'voting_code_sha256': RULE_SHA,
            'interpretation': 'Final reproduction on an already used Validation set. Not an independent test. Ensemble inference cost differs from a single model.'}


def validate_pro(destination):
    source = ROOT / 'scripts/audit_mmmu_pro_answers.py'
    assert source.is_file(), 'MMMU-Pro 감사 코드가 없습니다.'
    spec = importlib.util.spec_from_file_location('pro_final_records_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse the same audit logic and raw outputs; write into a new record directory.
    module.save = lambda path, value: save(destination / path.name, value)
    module.main()
    audit = read(destination / 'answer_format_audit.json')
    return {'status': 'completed', 'source_sha256': sha(source),
            'report': str((destination / 'answer_format_audit.json').relative_to(ROOT)),
            'interpretation': 'Saved-output parsing audit for base/R20. No new inference. Letter-only and content-verified results must be distinguished; original strict-parser scores are preserved.'}


def private_archive(destination, stamp):
    folders = [ROOT / 'scripts', ROOT / 'results_vllm_v2', FINAL,
               ROOT / 'results/independent_900_20261121', ROOT / 'results/mmmu_pro_base_R20_20261121',
               ROOT / 'logs/final_validation_900_20261121', ROOT / 'logs/final_validation_900_image_only_20261121',
               ROOT / 'logs/independent_900_20261121', ROOT / 'logs/soft_vote_pilot_20261121',
               ROOT / 'analysis_out/ensemble_exploration_20261121', ROOT / 'eval_candidates', destination]
    allowed = {'.py', '.sh', '.json', '.jsonl', '.md', '.txt', '.csv', '.log'}
    files = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in allowed and '__pycache__' not in path.parts and '.cache' not in path.parts:
                assert path.resolve().is_relative_to(ROOT.resolve()), f'프로젝트 밖의 파일입니다: {path}'
                files.add(path)
    files = sorted(files)
    archive = ROOT / 'backups' / ('mmdl_final_records_' + stamp + '.zip')
    archive.parent.mkdir(exist_ok=True)
    manifest = {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
        for index, path in enumerate(files):
            zipped.write(path, path.relative_to(ROOT).as_posix())
            if (index + 1) % 100 == 0:
                print('백업 파일:', index + 1, '/', len(files), flush=True)
        zipped.writestr('BACKUP_SHA256_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None, '압축 파일 내부 검사에 실패했습니다.'
    digest = sha(archive)
    archive.with_name(archive.name + '.sha256').write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
    return {'path': str(archive), 'files': len(files), 'bytes': archive.stat().st_size, 'sha256': digest,
            'scope': 'Private local code and result records only. Excludes model weights, virtual environments, raw images/Parquet, school-only files and optimizer checkpoints. Not a full school-server backup and not an automatically public Git upload.'}


def main(archive):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = ROOT / 'final_records' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    print('=== 최종 Validation 검증 ===', flush=True)
    validation = validate_final()
    save(destination / 'validation_check.json', validation)
    print(json.dumps(validation['results'], ensure_ascii=False, indent=2), flush=True)
    print('=== MMMU-Pro 저장 출력 감사 재현 ===', flush=True)
    pro = validate_pro(destination / 'mmmu_pro_audit')
    save(destination / 'pro_audit_check.json', pro)
    notes = '''# 최종 기록의 해석

- 과제 1의 순정 480/900은 보존합니다. 이번 같은 실행 조건의 비교 기준은 478/900입니다.
- 최종 Validation은 같은 900문항의 재현 평가이며 독립 검증이 아닙니다.
- 별도 Test900은 전체 Test10500과 구분합니다. Test 결과로 모델을 다시 선택하지 않습니다.
- MMMU-Pro는 기존 원시 출력의 답 형식 감사입니다. 답 문자 기준과 본문 검증 기준을 구분합니다.
- 단일 모델 학습 성과와 다중 모델 앙상블의 추론 성과를 구분합니다.
- 이 폴더는 새 기록입니다. 기존 평가 결과를 수정하지 않았습니다.
- ZIP은 개인 보관용 코드·결과 기록입니다. 모델·원본 데이터·학교 체크포인트 전체 백업은 아닙니다.
- Git 저장소 초기화·커밋·공개 업로드는 실행하지 않았습니다.
'''
    (destination / 'README.md').write_text(notes, encoding='utf-8')
    report = {'status': 'completed', 'records': str(destination), 'validation': validation['status'], 'pro_audit': pro['status'], 'new_gpu_inference': False}
    save(destination / 'completion.json', report)
    if archive:
        print('=== 개인 보관용 코드·결과 백업 ===', flush=True)
        report['backup'] = private_archive(destination, stamp)
        save(destination / 'backup_location.json', report['backup'])
    print('=== 최종 기록 정리 완료 ===', flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
    else:
        main(args.archive)
