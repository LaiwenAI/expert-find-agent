"""Audit profile-volume shifts between two frozen temporal rankings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import statistics
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    root = Path('artifacts/temporal-vote-vs-max-v1'); history = Path('artifacts/corpus-asof-2020-v1/evidence.jsonl')
    parent = Path('artifacts/local-temporal-author-proxy-v1/results.jsonl'); summary = json.loads((root / 'summary.json').read_text())
    if sha(root / 'results.json') != summary['results_sha256'] or sha(history) != summary['input_sha256']['history'] or sha(parent) != summary['input_sha256']['parent_results']:
        raise ValueError('Changed frozen data')
    cases = [json.loads(line) for line in parent.read_text().splitlines()]; results = json.loads((root / 'results.json').read_text())
    if len(cases) != 200 or [r['id'] for r in cases] != [r['id'] for r in results]: raise ValueError('Changed cohort')
    counts = Counter()
    for row in map(json.loads, history.read_text().splitlines()):
        if row['kind'] != 'paper': raise ValueError('Unexpected source type')
        counts[row['expert_id']] += 1
    measurements = []
    for row in results:
        values = {}
        for method in ('max', 'vote'):
            ranking = row['arms'][method]['ranking']
            if not ranking or len(ranking) != len(set(ranking)) or any(e not in counts for e in ranking): raise ValueError('Invalid ranking')
            volumes = [counts[e] for e in ranking]
            values[method] = {'mean_records': statistics.mean(volumes), 'median_records': statistics.median(volumes)}
        measurements.append(values)
    parents = list(range(len(cases)))
    def find(i):
        while parents[i] != i: parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owners = {}
    for i, case in enumerate(cases):
        for eid in case['known_author_ids']:
            if eid in owners: parents[find(i)] = find(owners[eid])
            else: owners[eid] = i
    groups = defaultdict(list)
    for i in range(len(cases)): groups[find(i)].append(i)
    groups = list(groups.values()); rng = random.Random(20261120); draws = []
    differences = [r['vote']['mean_records'] - r['max']['mean_records'] for r in measurements]
    for _ in range(10000):
        sample = [i for _ in groups for i in rng.choice(groups)]
        draws.append(sum(differences[i] for i in sample) / len(sample))
    draws.sort()
    report = {'queries': len(cases), 'experts_with_records': len(counts), 'historical_records': sum(counts.values()),
              'mean_per_query_expert_records': {method: statistics.mean(r[method]['mean_records'] for r in measurements) for method in ('max','vote')},
              'mean_per_query_median_records': {method: statistics.mean(r[method]['median_records'] for r in measurements) for method in ('max','vote')},
              'vote_minus_max_mean_records': statistics.mean(differences), 'cluster_bootstrap_percentile_95': [draws[249], draws[9749]],
              'queries_vote_higher': sum(x > 0 for x in differences), 'queries_vote_lower': sum(x < 0 for x in differences),
              'author_connected_groups': len(groups), 'bootstrap_replicates': 10000, 'seed': 20261120,
              'input_sha256': {'rankings': summary['results_sha256'], 'history': summary['input_sha256']['history'], 'parent': summary['input_sha256']['parent_results']},
              'code_sha256': sha(Path(__file__)),
              'scope': 'Post-hoc profile-volume audit of two saved rankings on 200 reused historical-title proxy queries. Record count is a property of the supplied snapshot; it is not verified publication count, personal contribution, quality, honorary authorship, or evidence support. This observational association is not causal. Bootstrap groups queries sharing known authors but not other topic or team dependencies; near-duplicates and incomplete authorship remain.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
