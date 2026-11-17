"""Replay a validation-selected fast/slow cascade on grouped MERIT test pairs."""
import hashlib
import json
import math
import statistics
from pathlib import Path
from expert_search.merit_suitability import prompts, build_case, gold
from experiments.analyze_merit_suitability import ROOT, DATA, bootstrap_cluster


def run():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    source = DATA / 'MERIT-Assessor/test/test.json'
    prompt_path = DATA / 'MERIT-Assessor/test/prompts.py'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(source) != protocol['dataset_sha256'] or digest(prompt_path) != protocol['official_prompts_sha256']:
        raise ValueError('Frozen source mismatch')
    rows = json.loads(source.read_text()); official = prompts(prompt_path)
    cases = [build_case(row, i, official) for i, row in enumerate(rows)]
    if len(cases) != 610:
        raise ValueError('Incomplete cohort')
    entries = []
    for case, raw in zip(cases, rows):
        predictions = {}
        for arm in ('official-long', 'native-choice'):
            path = ROOT / 'predictions' / (case['id'] + '-' + arm + '.json')
            value = json.loads(path.read_text())
            if value['input_sha256'] != case['input_sha256'] or value['case_id'] != case['id'] or value['arm'] != arm or value['status'] != 'valid' or value['label'] not in (0, 1) or not math.isfinite(value['seconds']) or value['seconds'] < 0:
                raise ValueError('Invalid frozen prediction')
            predictions[arm] = value
        answer = predictions['native-choice']['response']['answers'][0]
        probs = answer['probabilities']
        if len(probs) != 2 or {v['value'] for v in probs} != {'qualified', 'not_qualified'}:
            raise ValueError('Unexpected native choices')
        probability = max(v['probability'] for v in probs)
        if not .5 <= probability <= 1:
            raise ValueError('Invalid native scores')
        cluster = case['paper_cluster']
        # A fixed title hash keeps all exact-paper groups in the same partition.
        split = 'development' if int(hashlib.sha256(cluster.encode()).hexdigest()[:8], 16) % 2 == 0 else 'test'
        entries.append(dict(cluster=cluster, split=split, gold=gold(raw), probability=probability, predictions=predictions))
    def evaluate(items, threshold):
        flags = [x['probability'] < threshold for x in items]
        correct = []; times = []
        for x, flag in zip(items, flags):
            native = x['predictions']['native-choice']; full = x['predictions']['official-long']
            correct.append(int((full if flag else native)['label'] == x['gold']))
            times.append(native['seconds'] + (full['seconds'] if flag else 0))
        return dict(pairs=len(items), escalated=sum(flags), accuracy=statistics.mean(correct), simulated_mean_seconds=statistics.mean(times))
    dev = [e for e in entries if e['split'] == 'development']; test = [e for e in entries if e['split'] == 'test']
    thresholds = [0.5 + i * 0.05 for i in range(11)]
    budget = .7 * statistics.mean(x['predictions']['official-long']['seconds'] for x in dev)
    options = [(t, evaluate(dev, t)) for t in thresholds]
    admissible = [(t, r) for t, r in options if r['simulated_mean_seconds'] <= budget]
    if not admissible:
        raise ValueError('No threshold under development budget')
    chosen, development = max(admissible, key=lambda item: (item[1]['accuracy'], -item[1]['simulated_mean_seconds'], -item[0]))
    heldout = evaluate(test, chosen)
    diffs = []
    for x in test:
        a=x['predictions']['native-choice']; b=x['predictions']['official-long']
        chosen_prediction = b['label'] if x['probability'] < chosen else a['label']
        diffs.append(int(chosen_prediction == x['gold']) - int(b['label'] == x['gold']))
    report = dict(threshold=chosen, threshold_grid=thresholds, development_time_budget_seconds=budget,
                  development=development, test=heldout,
                  test_native_accuracy=statistics.mean(x['predictions']['native-choice']['label']==x['gold'] for x in test),
                  test_full_accuracy=statistics.mean(x['predictions']['official-long']['label']==x['gold'] for x in test),
                  test_full_mean_seconds=statistics.mean(x['predictions']['official-long']['seconds'] for x in test),
                  test_accuracy_difference_vs_full=bootstrap_cluster(diffs,[x['cluster'] for x in test]),
                  source_sha256=digest(source), protocol_sha256=digest(ROOT / 'protocol.json'),
                  scope='Post-hoc offline replay of existing calls. Threshold selected on development self-ratings under a 70% full-request time budget; test grouped by exact paper title. Request times were recorded under prior interleaving and concurrency, not an actual serial cascade. Public self-ratings are not enterprise expert-fit gold; confidence is an uncalibrated model score. Exact titles do not guarantee semantic or reviewer independence.')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    run()
