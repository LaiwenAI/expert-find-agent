"""Paired binary versus factored native Decisions on complete MERIT610 inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from expert_search.choice_judgments import decide
from expert_search.factored_suitability import questions, combine
from expert_search.merit_suitability import prompts, build_case

DATA = Path('data/public/merit-release-20261117')
ROOT = Path('artifacts/merit-factored-comparison-v1')
ARMS = ['native-binary', 'native-factored']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    baseline = Path('artifacts/merit-suitability-comparison-v1/protocol.json')
    old = json.loads(baseline.read_text())
    source, prompt = DATA / 'MERIT-Assessor/test/test.json', DATA / 'MERIT-Assessor/test/prompts.py'
    if sha(source) != old['dataset_sha256'] or sha(prompt) != old['official_prompts_sha256']:
        raise ValueError('Changed official benchmark')
    official = prompts(prompt)
    cases = [build_case(row, index, official) for index, row in enumerate(json.loads(source.read_text()))]
    if len(cases) != 610:
        raise ValueError('Incomplete released input cohort')
    protocol = dict(model='gpt-6-luna', cases=610, dataset_sha256=sha(source), official_prompts_sha256=sha(prompt), inherited_criterion=old['criterion'], binary_choices=old['choices'], factored_questions=questions(old['criterion']), arms=ARMS, original_baseline_protocol_sha256=sha(baseline), order='Alternate two arm orders over original610case order;3workers,no retries. Re-run binary contemporaneously; original long arm is historical only.', aggregation='Three-valued conjunction: any not_demonstrated =>0; both demonstrated =>1; otherwise abstain. No learned threshold.', code_sha256={str(p): sha(p) for p in [Path(__file__), Path('src/expert_search/factored_suitability.py'), Path('src/expert_search/choice_judgments.py'), Path('src/expert_search/merit_suitability.py')]}, scope='Exploratory task-rule factorization after observing previous benchmark aggregates, not pristine held-out evaluation or trained MERIT reproduction. Labels excluded from inference. Self-rated reviewer expertise is auxiliary, not enterprise transfer gold. No model training, explanation generation or online audit stage.')
    ROOT.mkdir(exist_ok=True)
    marker = ROOT / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed frozen factorization protocol')
    marker.write_text(json.dumps(protocol, indent=2) + '\n')
    (ROOT / 'predictions').mkdir(exist_ok=True)
    return cases, protocol


def one(item, protocol):
    index, case = item
    from laiwenai import LaiwenAI
    client = LaiwenAI(model='gpt-6-luna', max_retries=0)
    output = {}
    for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
        path = ROOT / 'predictions' / (case['id'] + '-' + arm + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            if value['input_sha256'] != case['input_sha256'] or value['arm'] != arm:
                raise ValueError('Changed cached input')
            output[arm] = value
            continue
        start = time.perf_counter()
        value = dict(case_id=case['id'], arm=arm, input_sha256=case['input_sha256'], paper_cluster=case['paper_cluster'], status='unresolved', label=None)
        try:
            qs = protocol['factored_questions'] if arm == 'native-factored' else [dict(type='choice', name='suitability', instructions=protocol['inherited_criterion'], choices=protocol['binary_choices'])]
            response = decide(client, dict(target_and_profile=case['input']), qs)
            value['response'] = response['response']
            value['primitive_labels'] = response['labels']
            label = combine(response['labels']) if arm == 'native-factored' else int(response['labels']['suitability'] == 'qualified')
            value.update(label=label, status='abstained' if label is None else 'valid')
        except Exception as error:
            value.update(error_type=type(error).__name__, http_status=getattr(error, 'status_code', None))
        value['seconds'] = time.perf_counter() - start
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
        output[arm] = value
    print(case['id'], {arm: value['status'] for arm, value in output.items()}, flush=True)
    return output


def run():
    cases, protocol = prepare()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda item: one(item, protocol), enumerate(cases)))
    report = dict(terminal_pairs=len(rows), statuses={a: {s: sum(r[a]['status'] == s for r in rows) for s in ['valid', 'abstained', 'unresolved']} for a in ARMS}, scope='Progress only; no partial accuracy. All610pairs required before evaluation.')
    (ROOT / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    run()
