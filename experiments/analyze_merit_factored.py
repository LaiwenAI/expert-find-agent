"""Complete610paired evaluation; keep abstentions separate from failed requests."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from expert_search.factored_suitability import combine
from expert_search.merit_suitability import prompts, build_case, gold
from experiments.analyze_merit_suitability import metrics, bootstrap_cluster

ROOT = Path('artifacts/merit-factored-comparison-v1')
DATA = Path('data/public/merit-release-20261117')
ARMS = ['native-binary', 'native-factored']


def run():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    source, prompt = DATA / 'MERIT-Assessor/test/test.json', DATA / 'MERIT-Assessor/test/prompts.py'
    for path, expected in [(source, protocol['dataset_sha256']), (prompt, protocol['official_prompts_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed official input')
    raw = json.loads(source.read_text())
    cases = [build_case(row, index, prompts(prompt)) for index, row in enumerate(raw)]
    labels = [gold(row) for row in raw]
    if len(cases) != 610:
        raise ValueError('Invalid released cohort')
    predictions = {a: [] for a in ARMS}
    hashes = {}
    for case in cases:
        for arm in ARMS:
            path = ROOT / 'predictions' / (case['id'] + '-' + arm + '.json')
            if not path.exists():
                raise ValueError('Refuse partial factorization run')
            value = json.loads(path.read_text())
            if value['input_sha256'] != case['input_sha256'] or value['case_id'] != case['id'] or value['arm'] != arm or value['paper_cluster'] != case['paper_cluster']:
                raise ValueError('Prediction provenance mismatch')
            if value['status'] not in ('valid', 'abstained', 'unresolved') or not math.isfinite(value['seconds']) or value['seconds'] < 0:
                raise ValueError('Invalid state/time')
            if value['status'] == 'valid':
                if type(value['label']) is not int or value['label'] not in (0, 1):
                    raise ValueError('Invalid class')
            elif value['label'] is not None:
                raise ValueError('Unresolved label imputation')
            if value['status'] != 'unresolved':
                derived = combine(value['primitive_labels']) if arm == 'native-factored' else int(value['primitive_labels']['suitability'] == 'qualified')
                if derived != value['label'] or (derived is None) != (value['status'] == 'abstained'):
                    raise ValueError('Primitive/conjunction disagreement')
            predictions[arm].append(value)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    arms = {}
    for arm, values in predictions.items():
        times = [v['seconds'] for v in values]
        arms[arm] = dict(metrics(labels, [v['label'] for v in values]), statuses={s: sum(v['status'] == s for v in values) for s in ['valid', 'abstained', 'unresolved']}, mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)), empirical_p95_seconds=float(np.quantile(times, .95)))
    first, second = predictions['native-factored'], predictions['native-binary']
    valid = [i for i in range(610) if first[i]['status'] == 'valid' and second[i]['status'] == 'valid']
    contrasts = dict(common_valid_pairs=len(valid), accuracy_difference_common_valid=bootstrap_cluster([int(first[i]['label'] == labels[i]) - int(second[i]['label'] == labels[i]) for i in valid], [cases[i]['paper_cluster'] for i in valid]) if valid else None, time_difference_all_attempts=bootstrap_cluster([a['seconds'] - b['seconds'] for a, b in zip(first, second)], [r['paper_cluster'] for r in cases]), full_cohort_accuracy_difference_bounds=[arms['native-factored']['full_cohort_bounds']['accuracy'][0] - arms['native-binary']['full_cohort_bounds']['accuracy'][1], arms['native-factored']['full_cohort_bounds']['accuracy'][1] - arms['native-binary']['full_cohort_bounds']['accuracy'][0]])
    report = dict(pairs=610, positive_ratings=sum(labels), exact_target_paper_clusters=len({r['paper_cluster'] for r in cases}), arms=arms, factored_minus_binary=contrasts, protocol_sha256=hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest(), predictions_sha256=hashes, code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('experiments/analyze_merit_suitability.py'), Path('src/expert_search/factored_suitability.py')]}, scope=protocol['scope'] + ' Common-valid accuracy is conditional on joint non-abstention; full-cohort bounds and coverage must accompany it. Probability values are not calibrated confidence. No threshold fitting or hidden-label input; endpoint nondeterminism and prior aggregate-driven exploration remain limitations.')
    (ROOT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('predictions_sha256', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    run()
