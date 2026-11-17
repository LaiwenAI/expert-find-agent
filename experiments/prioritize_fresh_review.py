"""Prioritize fixed-view reviews by missing-label bound reduction."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from expert_search.evaluation_gain import actionable_gain_interval, paired_actionable_gain_interval


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); base = Path('artifacts/fresh-evaluation-baselines-v1')
    manifest = json.loads((data / 'manifest.json').read_text()); summary = json.loads((base / 'summary.json').read_text())
    if sha(data / 'manifest.json') != summary['dataset_manifest_sha256']: raise ValueError('Changed dataset')
    for name in ('queries.jsonl', 'pools.jsonl', 'judgments.jsonl', 'snapshots.jsonl'):
        if sha(data / name) != manifest['files_sha256'][name]: raise ValueError('Changed data member')
    queries = {r['id'] for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development'}
    pools = {r['query_id']: r['candidates'] for r in map(json.loads, (data / 'pools.jsonl').read_text().splitlines()) if r['query_id'] in queries}
    grades = defaultdict(dict); reasons = {}
    for row in map(json.loads, (data / 'judgments.jsonl').read_text().splitlines()):
        qid, eid = row['query_id'], row['expert_id']
        if qid in queries:
            if eid in grades[qid]: raise ValueError('Duplicate grade')
            grades[qid][eid] = row['grade']; reasons[(qid, eid)] = row['reason']
    rankings = defaultdict(dict)
    for row in json.loads((base / 'evaluation.json').read_text()):
        qid, method = row['query_id'], row['method']
        if qid not in queries or method in rankings[qid] or len(row['ranking']) != len(set(row['ranking'])) or set(row['ranking']) != set(pools[qid]): raise ValueError('Invalid ranking')
        if actionable_gain_interval(row['ranking'], grades[qid], 5) != row['actionable_gain']: raise ValueError('Changed evaluation')
        rankings[qid][method] = row['ranking']
    if set(rankings) != queries: raise ValueError('Missing queries')
    weights = [1 / math.log2(i + 2) for i in range(5)]; normalizer = sum(weights); choices = []; original_width = 0.
    for qid in sorted(queries):
        a, b = rankings[qid]['qwen-paper-max'][:5], rankings[qid]['bm25-max'][:5]
        coeff = defaultdict(float)
        for sign, ranking in ((1, a), (-1, b)):
            for rank, eid in enumerate(ranking): coeff[eid] += sign * weights[rank] / normalizer
        before = paired_actionable_gain_interval(a, b, grades[qid], 5)
        original_width += before['upper'] - before['lower']
        for eid, value in coeff.items():
            if grades[qid].get(eid) is not None or abs(value) < 1e-15: continue
            trial = dict(grades[qid]); trial[eid] = 0
            after = paired_actionable_gain_interval(a, b, trial, 5)
            if not math.isclose(before['upper']-before['lower']-after['upper']+after['lower'], abs(value), abs_tol=1e-12):
                raise ValueError('Unexpected bound reduction')
            choices.append({'query_id': qid, 'expert_id': eid, 'reason': reasons[(qid, eid)], 'bound_width_reduction': abs(value) / len(queries)})
    choices.sort(key=lambda r: (-r['bound_width_reduction'], r['query_id'], r['expert_id']))
    width = original_width / len(queries)
    report = {'queries': len(queries), 'eligible_unjudged_views': len(choices), 'mean_interval_width': width,
              'potential_reduction': {str(n): sum(r['bound_width_reduction'] for r in choices[:n]) for n in (10, 25, 50, 100)},
              'reason_counts': {reason: sum(r['reason'] == reason for r in choices) for reason in sorted({r['reason'] for r in choices})},
              'input_sha256': {'dataset_manifest': sha(data / 'manifest.json'), 'baseline_evaluation': sha(base / 'evaluation.json'), 'baseline_summary': sha(base / 'summary.json')},
              'scope': 'Post-hoc review triage for Qwen paper-max minus BM25-max on the same fixed development views. Each potential reduction assumes an independent reviewer supplies a definite actionable/nonactionable label on that exact view; labels, reviewer time, errors and semantic quality are not estimated. Ranking by influence targets this comparison, not representative dataset sampling, full-corpus accuracy or held-out evaluation. IDs and source text stay local.'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'private-priority.json').write_text(json.dumps(choices, indent=2) + '\n')
    report['priority_sha256'] = sha(output / 'private-priority.json'); report['code_sha256'] = sha(Path(__file__))
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
