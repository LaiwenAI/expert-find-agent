"""Paired per-query quality and timing tradeoffs for the saved CoF diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('artifacts/cof-sigir-adaptive-scores-v1')


def analyze(rows, repetitions=10000, seed=20261118):
    if not rows or len({row['query_id'] for row in rows}) != len(rows):
        raise ValueError('Missing or duplicated queries')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    comparisons = {}
    for method in ('adaptive20', 'adaptive50'):
        candidate = [r['methods'][method] for r in rows]
        baseline = [r['methods']['global'] for r in rows]
        timing = np.array([a['seconds'] - b['seconds'] for a, b in zip(candidate, baseline)])
        if not np.isfinite(timing).all():
            raise ValueError('Invalid timings')
        quality = {}
        for task in ('soft', 'hard'):
            for metric in ('p@5', 'ndcg@10'):
                delta = np.array([a['metrics'][task][metric] - b['metrics'][task][metric] for a, b in zip(candidate, baseline)])
                if not np.isfinite(delta).all():
                    raise ValueError('Invalid quality values')
                quality[f'{task}:{metric}'] = dict(mean_difference=float(delta.mean()),
                    percentile_95=list(map(float, np.quantile(delta[draws].mean(axis=1), [.025, .975]))),
                    improved=int(np.sum(delta > 1e-12)), tied=int(np.sum(np.abs(delta) <= 1e-12)),
                    worsened=int(np.sum(delta < -1e-12)))
        comparisons[method] = dict(quality=quality, mean_seconds_difference=float(timing.mean()),
                                   faster_queries=int(np.sum(timing < 0)))
    return dict(queries=len(rows), bootstrap_repetitions=repetitions, seed=seed, comparisons=comparisons,
                scope='Post-hoc paired query bootstrap on one CoF SIGIR test cohort after inspecting results; intervals assume independently sampled queries and do not correct for multiple comparisons. Reviewer-matching labels are not industrial expert-fit judgments. Single timing per query excludes indexing, initialization, evidence display and reasoning; faster-query counts do not establish latency distributions or deployment speed.')


def main():
    path = ROOT / 'results.json'
    raw = path.read_bytes()
    rows = json.loads(raw)
    report = analyze(rows)
    summary = json.loads((ROOT / 'summary.json').read_text())
    if report['queries'] != summary['queries']:
        raise ValueError('Results do not match summary query count')
    for method in ('adaptive20', 'adaptive50', 'global'):
        for task in ('soft', 'hard'):
            for metric in ('p@5', 'ndcg@10'):
                mean = np.mean([r['methods'][method]['metrics'][task][metric] for r in rows])
                if not np.isclose(mean, summary['metrics'][task][method][metric], atol=1e-12):
                    raise ValueError('Results do not match summary metrics')
    report['results_sha256'] = hashlib.sha256(raw).hexdigest()
    report['summary_sha256'] = hashlib.sha256((ROOT / 'summary.json').read_bytes()).hexdigest()
    report['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = ROOT / 'paired-tradeoff.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
