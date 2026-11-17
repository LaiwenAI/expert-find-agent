"""Run order-controlled single-view grounding checks with frozen inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import argparse
from pathlib import Path
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_disagreement import CHOICES
from .grounding_control_cases import cases, GROUNDING_INSTRUCTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross-language', action='store_true')
    parser.add_argument('--chinese', action='store_true')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--repeat-real', action='store_true')
    args = parser.parse_args()
    if args.repeat_real and not args.real:
        parser.error('--repeat-real requires --real')
    if sum((args.cross_language, args.chinese, args.real)) > 1:
        parser.error('Choose one control cohort')
    selected = cases()
    if args.chinese:
        from .grounding_control_cases import chinese_cases
        selected = chinese_cases()
    if args.cross_language:
        from .audit_order_controls import synthetic_cases
        selected = [dict(id='cross-' + c['id'], source=c['payload']['original_need'],
                         derived=c['payload']['en'], expected=c['expected']) for c in synthetic_cases()]
    if args.real:
        source = Path('artifacts/bilingual-order-controls-v1/cases.jsonl')
        original = [json.loads(line) for line in source.read_text().splitlines()]
        selected = [dict(id=c['id'] + ':' + lang, source=c['payload']['original_need'],
                         derived=c['payload'][lang], expected=None)
                    for c in original if c['group'] == 'real' for lang in ('zh', 'en')]
        if len(selected) != 60:
            raise ValueError('Expected 30 real paired cases')
    root = Path('artifacts/grounding-real-repeat-v1' if args.repeat_real else 'artifacts/grounding-real-v1' if args.real else 'artifacts/grounding-chinese-v1' if args.chinese else 'artifacts/grounding-cross-language-v1' if args.cross_language else 'artifacts/grounding-single-view-v1')
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for case in selected:
        for order in ([(0, 1, 2), (2, 1, 0)] if args.real else itertools.permutations(range(3))):
            payload = {k: case[k] for k in ('source', 'derived')}
            questions = [dict(type='choice', name='grounding', instructions=GROUNDING_INSTRUCTION,
                              choices=[CHOICES[i] for i in order])]
            identity = dict(case_id=case['id'], order=order, payload=payload, questions=questions)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            tasks.append((case, identity, key))
    protocol = dict(model='gpt-6-luna', tasks=len(tasks), instructions=GROUNDING_INSTRUCTION,
                    cases=selected, choices=CHOICES)
    marker = root / 'protocol.json'
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + '\n'
    if marker.exists() and marker.read_text() != encoded:
        raise ValueError('Frozen protocol differs')
    marker.write_text(encoded)
    (root / 'predictions').mkdir(exist_ok=True)
    load_dotenv('.env')

    def run(item):
        case, identity, key = item
        path = root / 'predictions' / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())
        start = time.perf_counter()
        record = dict(key=key, case_id=case['id'], order=identity['order'], expected=case['expected'])
        try:
            result = decide(LaiwenAI(model='gpt-6-luna', max_retries=0), identity['payload'], identity['questions'])
            record.update(status='valid', prediction=result['labels']['grounding'], response=result['response'])
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__)
        record['seconds'] = time.perf_counter() - start
        tmp = path.with_suffix('.partial')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(path)
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, tasks))
    valid = [r for r in results if r['status'] == 'valid']
    summary = dict(total=len(results), valid=len(valid), failed=len(results)-len(valid),
                   correct=None if args.real else sum(r['prediction'] == r['expected'] for r in valid),
                   scope='Real cases are unjudged; synthetic control labels excluded from model input; no relevance or generalization claim.')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
