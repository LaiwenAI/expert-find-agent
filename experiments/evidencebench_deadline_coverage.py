"""Replay time-integrated labeled evidence coverage at fixed horizons."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.evaluation_gain import time_integrated_gain_interval
from expert_search.evidencebench_data import prepare, aspect_coverage
from expert_search.statistics import paired_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    native = Path('artifacts/evidencebench-decisions-v1'); sets = Path('artifacts/evidencebench-set-selector-v1')
    native_protocol = json.loads((native / 'protocol.json').read_text()); protocol = json.loads((sets / 'protocol.json').read_text())
    data = prepare()
    if sha(Path('artifacts/public/evidencebench/dev.json')) != native_protocol['dataset_sha256'] or set(protocol['parents']) != {k + '.json' for k in data}:
        raise ValueError('Changed data or cohort')
    rows = []
    for key, source in data.items():
        path = native / (key + '.json')
        if sha(path) != protocol['parents'][path.name]: raise ValueError('Changed native control')
        a = json.loads(path.read_text()); b = json.loads((sets / path.name).read_text())
        if a['id'] != key or b['id'] != key or a['status'] != 'valid' or aspect_coverage(source, a['selected']) != a['metrics']:
            raise ValueError('Invalid baseline')
        if b['status'] == 'valid':
            if aspect_coverage(source, b['selected']) != b['metrics'] or not set(b['selected']) <= {int(i[1:]) for i in a['decision']['scores']}:
                raise ValueError('Changed selected evidence')
        elif b['metrics'] != a['same_query_baseline']:
            raise ValueError('Changed failure fallback')
        times = {'native': a['coarse_seconds'] + a['api_wall_seconds'], 'set': a['coarse_seconds'] + b['request_seconds']}
        coverage = {'native': a['metrics']['aspect_recall'], 'set': b['metrics']['aspect_recall']}
        gains = {str(h): {arm: time_integrated_gain_interval([(times[arm], coverage[arm], coverage[arm])], h)['lower'] for arm in times} for h in (5, 10)}
        rows.append({'id': key, 'set_valid': b['status'] == 'valid', 'times': times, 'coverage': coverage, 'gains': gains})
    report = {'cases': len(rows), 'set_valid': sum(r['set_valid'] for r in rows), 'horizons': {}}
    for horizon in (5, 10):
        h = str(horizon)
        report['horizons'][h] = {'completed': {a: sum(r['times'][a] <= horizon for r in rows) for a in ('native','set')},
                                 'mean_integrated_coverage': {a: statistics.mean(r['gains'][h][a] for r in rows) for a in ('native','set')},
                                 'paired_set_minus_native': paired_interval([r['gains'][h]['set'] - r['gains'][h]['native'] for r in rows])}
    report['input_sha256'] = {'native_protocol': sha(native / 'protocol.json'), 'set_protocol': sha(sets / 'protocol.json'),
                              'set_records': {r['id']: sha(sets / (r['id'] + '.json')) for r in rows}}
    report['scope'] = 'Post-hoc 5/10-second horizon replay of 37 reused EvidenceBench development cases, not expert-fit or SetR reproduction. Reuses the native recorded query encoding time as shared coarse latency for both arms, then adds historical nonrandomized request times; two failed set calls return the semantic fallback only after failure. Omits initialization, index construction, queueing outside recorded calls and other overhead. Labeled aspect coverage is not evidence entailment or a system SLA; intervals assume independent cases and are not multiple-comparison adjusted.'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = sha(output / 'results.json'); report['code_sha256'] = sha(Path(__file__))
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps({k:v for k,v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
