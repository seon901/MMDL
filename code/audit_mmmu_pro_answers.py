from pathlib import Path
from collections import Counter
import ast
import json
import math
import re
import unicodedata

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / 'results/mmmu_pro_base_R20_20261121'
DATA = PROJECT / 'evaluation_reference/MMMU_Pro/0d7426df4ccb3d8704a992fe6850abee7c262127'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())

def unwrap(value):
    value = value.strip()
    for wrapper in ['**', '__', '`']:
        if value.startswith(wrapper) and value.endswith(wrapper) and len(value) > 2*len(wrapper):
            value = value[len(wrapper):-len(wrapper)].strip()
    return value

def parse(raw, choices):
    text = unwrap(raw)
    explicit = re.findall(r'^\s*(?:\*\*)?Answer\s*:\s*(.*?)\s*$', text, flags=re.I | re.M)
    candidate = unwrap(explicit[-1]) if explicit else text
    exact = [chr(65+i) for i, choice in enumerate(choices) if norm(candidate) == norm(choice)]
    if not explicit and len(exact) == 1 and not re.fullmatch(r'\(?[A-Z]\)?[.]?', candidate, flags=re.I):
        return {'letter': exact[0], 'verified': exact[0], 'status': 'exact_option_text'}
    match = re.fullmatch(r'(?:\(([A-Z])\)|([A-Z])(?:[.):\]]|(?=\s|$)))\s*(.*)', candidate, flags=re.I | re.S)
    if not match:
        return {'letter': None, 'verified': None, 'status': 'unresolved'}
    letter = (match.group(1) or match.group(2)).upper()
    if not 0 <= ord(letter)-65 < len(choices):
        return {'letter': None, 'verified': None, 'status': 'invalid_letter'}
    tail = unwrap(match.group(3))
    if not tail:
        return {'letter': letter, 'verified': letter, 'status': 'letter_only'}
    matching = [chr(65+i) for i, choice in enumerate(choices) if norm(tail) == norm(choice)]
    if matching == [letter]:
        return {'letter': letter, 'verified': letter, 'status': 'letter_and_option_agree'}
    if matching and letter not in matching:
        return {'letter': letter, 'verified': None, 'status': 'letter_option_conflict'}
    return {'letter': letter, 'verified': None, 'status': 'unverified_tail'}

def paired(before, after):
    recovered = sum(not a and b for a,b in zip(before,after))
    regressed = sum(a and not b for a,b in zip(before,after))
    n = recovered+regressed
    p = min(1.0, 2*sum(math.comb(n,k) for k in range(min(recovered,regressed)+1))/(2**n)) if n else 1.0
    return {'회복': recovered, '퇴보': regressed, '정답 순증가': recovered-regressed, 'McNemar exact p': p}

def main():
    import pyarrow.parquet as pq
    report = {}
    all_details = {}
    for setting, config in [('standard','standard (10 options)'), ('vision','vision')]:
        originals = {}
        for path in sorted((DATA/config).glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['id','options','answer']):
                for row in batch.to_pylist():
                    choices = ast.literal_eval(row['options']) if isinstance(row['options'],str) else row['options']
                    assert row['id'] not in originals
                    originals[row['id']] = (choices,row['answer'])
        assert len(originals)==1730
        models = {}
        details = {}
        for name in ['base','R20']:
            values = read(RESULTS/name/(setting+'_predictions.json'))
            assert len(values)==1730 and len({r['id'] for r in values})==1730
            parsed = []
            for row in values:
                choices,gold = originals[row['id']]
                assert row['gold']==gold and row['option_count']==len(choices)
                display_choices = [re.sub(r'<image\s+(\d+)>', lambda m: 'Figure ' + m.group(1), value) for value in choices] if setting == 'standard' else choices
                answer = parse(row['raw'], display_choices)
                parsed.append({**row, 'audit': answer, 'letter_correct': answer['letter']==gold, 'verified_correct': answer['verified']==gold})
            models[name] = {'total':len(values), 'original_correct':sum(r['correct'] for r in values),
                            'letter_correct':sum(r['letter_correct'] for r in parsed),
                            'verified_correct':sum(r['verified_correct'] for r in parsed),
                            'status_counts':dict(Counter(r['audit']['status'] for r in parsed)),
                            'newly_verified_correct':sum(not r['correct'] and r['verified_correct'] for r in parsed)}
            details[name] = parsed
        assert all(a['id']==b['id'] and a['input_sha256']==b['input_sha256'] for a,b in zip(details['base'],details['R20']))
        report[setting] = {**models, 'letter_paired':paired([r['letter_correct'] for r in details['base']],[r['letter_correct'] for r in details['R20']]),
                          'verified_paired':paired([r['verified_correct'] for r in details['base']],[r['verified_correct'] for r in details['R20']])}
        all_details[setting] = details
    save(RESULTS/'answer_format_audit.json', {'결과':report,
         'method':'Separate audit of saved outputs. Explicit answer line or a whole response beginning with a choice label. Choice text comparison preserves case and punctuation; whitespace is normalized. Standard image references use the same Figure labels as the inference input. Conflicts and unmatched tails are not verified.',
         'limitation':'Verified scores are conservative counts, not fully adjudicated semantic accuracy. Letter counts are a separate sensitivity analysis. Original results, generation and evaluator remain unchanged.'})
    save(RESULTS/'answer_format_audit_details.json', all_details)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
