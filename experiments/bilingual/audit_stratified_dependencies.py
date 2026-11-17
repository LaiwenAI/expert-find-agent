"""Audit cluster dependence of a frozen, screen-blind review sample."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sample = Path('artifacts/stratified-grounding-human-review-v1')
    data = Path('artifacts/dataset-release/bilingual-query-construction-v1')
    sm = json.loads((sample / 'manifest.json').read_text())
    dm = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'manifest.json') != sm['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    for name, digest in dm['outputs_sha256'].items():
        if sha(data / name) != digest:
            raise ValueError('Dataset output changed')
    if 'private_mapping_sha256' in sm and sha(sample / 'private-mapping.json') != sm['private_mapping_sha256']:
        raise ValueError('Review mapping changed')
    mapping = json.loads((sample / 'private-mapping.json').read_text())
    if len(mapping) != sm['cases'] or len({r['case_id'] for r in mapping}) != len(mapping):
        raise ValueError('Invalid cases')
    parents = {r['id']: r for r in map(json.loads, (data / 'real-queries.jsonl').read_text().splitlines())}
    rows = {r['id']: r for r in map(json.loads, (data / 'synthetic-queries.jsonl').read_text().splitlines())}
    strata = defaultdict(list)
    for row in rows.values():
        strata[row['intent_type'], row['status']].append(row)
    rng = random.Random(sm['seed'])
    selected = []
    for spec in sm['strata']:
        key = (spec['intent_type'], spec['status'])
        population = sorted(strata[key], key=lambda row: row['id'])
        if len(population) != spec['population']:
            raise ValueError('Stratum population changed')
        selected.extend(rng.sample(population, spec['sample']))
    rng.shuffle(selected)
    expected = [{'case_id': f'case-{i:03d}', 'query_id': row['id'],
                 'parent_id': row['parent_id'], 'status': row['status']}
                for i, row in enumerate(selected, 1)]
    if mapping != expected:
        raise ValueError('Sample mapping differs from deterministic reconstruction')
    group_sizes, parent_sizes = Counter(), Counter()
    group_strata = defaultdict(set)
    for item in mapping:
        row = rows[item['query_id']]
        parent = parents[item['parent_id']]
        if row['parent_id'] != item['parent_id'] or row['status'] != item['status'] or parent['split'] != 'construction':
            raise ValueError('Sample mapping inconsistent')
        group = parent['group_id']
        if not group:
            raise ValueError('Missing parent group')
        parent_sizes[item['parent_id']] += 1
        group_sizes[group] += 1
        group_strata[group].add((row['intent_type'], row['status']))
    def summary(counts):
        hist = Counter(counts.values())
        return {'clusters': len(counts), 'size_histogram': {str(k): v for k, v in sorted(hist.items())},
                'cases_in_repeated_clusters': sum(n for n in counts.values() if n > 1),
                'largest_cluster': max(counts.values(), default=0)}
    result = {'cases': len(mapping), 'parents': summary(parent_sizes), 'groups': summary(group_sizes),
              'groups_spanning_multiple_strata': sum(len(v) > 1 for v in group_strata.values()),
              'input_sha256': {'sample_manifest': sha(sample / 'manifest.json'), 'dataset_manifest': sha(data / 'manifest.json'),
                               'private_mapping': sha(sample / 'private-mapping.json'), 'script': sha(Path(__file__))},
              'scope': 'Screen-blind stratified sample of construction queries only. Group IDs reflect company/body/URL grouping, not all possible semantic, author or topic dependencies. This diagnoses sample dependence and cannot estimate quality or replace independent human annotation. No text or identifiers are printed.'}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
