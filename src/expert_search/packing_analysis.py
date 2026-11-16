"""Paired analysis of frozen packing experiments; never silently omit failed calls."""
import itertools
import json
from pathlib import Path
import numpy as np
from .cof_benchmark import load, precision_metrics
from .cof_packing_experiment import CONDITIONS


from .statistics import paired_interval


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid measured values')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def analyze(dataset='SIGIR', serialization='inline'):
    if serialization not in ('inline', 'shared'):
        raise ValueError('Invalid serialization')
    root, _, _, labels = load(dataset)
    folder = root / ('packing-shared-v2' if serialization == 'shared' else 'packing-v1')
    protocol = json.loads((folder / 'protocol.json').read_text())
    qids = protocol['all_query_ids']
    rows = {}
    missing = []
    for qid, condition in itertools.product(qids, CONDITIONS):
        path = folder / f'{qid}-{condition}.json'
        if not path.exists():
            missing.append([qid, condition])
            continue
        row = json.loads(path.read_text())
        if row['query_id'] != qid or row['condition'] != condition:
            raise ValueError('Mislabeled experiment record')
        aliases, ranking, coarse = row['aliases'], row['ranking'], row['coarse_ranking']
        if len(set(aliases)) != len(aliases) or sorted(aliases) != sorted(coarse) or sorted(ranking) != sorted(aliases):
            raise ValueError('Invalid candidate permutation')
        if row['status'] not in ('ok', 'error') or (row['status'] == 'error' and ranking != coarse):
            raise ValueError('Failure must retain coarse fallback')
        for task in labels:
            if row['metrics'][task] != precision_metrics(ranking, labels[task][qid]):
                raise ValueError('Stored metrics differ from source labels')
        rows[qid, condition] = row
    # Incomplete runs produce a status, not a selected-subset comparison.
    if missing:
        return dict(dataset=dataset, complete=False, completed=len(rows), planned=len(qids)*3, missing=missing)
    for qid in qids:
        if any(rows[qid, c]['aliases'] != rows[qid, 'full']['aliases'] or
               rows[qid, c]['coarse_ranking'] != rows[qid, 'full']['coarse_ranking'] for c in CONDITIONS):
            raise ValueError('Conditions do not share the candidate pool and aliases')
    summaries = {}
    for condition in CONDITIONS:
        selected = [rows[qid, condition] for qid in qids]
        summaries[condition] = dict(
            failures=[r['query_id'] for r in selected if r['status'] != 'ok'],
            metrics={task: {key: float(np.mean([r['metrics'][task][key] for r in selected]))
                            for key in selected[0]['metrics'][task]} for task in labels},
            measurements={key: distribution([r.get(key, 0.) for r in selected]) for key in
                          ('model_call_seconds', 'rate_wait_seconds', 'preparation_seconds',
                           'evidence_characters', 'transmitted_evidence_characters')},
            prompt_tokens=distribution([r['usage'].get('prompt_tokens', 0) for r in selected]),
            http_status_counts={str(status): sum(a.get('status') == status for r in selected for a in r.get('http_attempts', []))
                                for status in sorted({a.get('status') for r in selected for a in r.get('http_attempts', []) if a.get('status') is not None})})
    contrasts = {}
    for first, second in itertools.combinations(CONDITIONS, 2):
        contrasts[f'{second}-minus-{first}'] = {task: {metric: paired_interval([
            rows[qid, second]['metrics'][task][metric] - rows[qid, first]['metrics'][task][metric]
            for qid in qids]) for metric in ('p@5', 'p@10')} for task in labels}
    return dict(dataset=dataset, serialization=serialization, complete=True, queries=len(qids),
                conditions=summaries, paired_quality_differences=contrasts,
                limitations='Query bootstrap is exploratory, assumes independent queries, and is not corrected for multiple comparisons. An interval crossing zero does not establish equivalence or noninferiority. Failed calls use coarse fallback and stay in every denominator. Timing excludes query encoding, initialization and rate waiting; report waits separately. Inline cyclic order permits carryover; shared order balances six permutations. Neither is a controlled load test or end-to-end SLA.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['SIGIR', 'NIPS'], default='SIGIR')
    parser.add_argument('--serialization', choices=['inline', 'shared'], default='inline')
    args = parser.parse_args()
    report = analyze(args.dataset, args.serialization)
    destination = Path('docs/exploration') / f'cof-{args.dataset.lower()}-packing-{args.serialization}-analysis.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(destination), complete=report['complete']), indent=2))
