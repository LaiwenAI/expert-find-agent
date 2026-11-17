"""Audit numeric-anchor coverage of an existing blind review sample."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Checksum mismatch: ' + str(path))
    return data


def run():
    data = Path('artifacts/dataset-release/bilingual-query-construction-v1')
    review = Path('artifacts/stratified-grounding-human-review-v1')
    post = Path('artifacts/large-bilingual-data-v1/postvalidation-v2')
    dataset_manifest = json.loads((data / 'manifest.json').read_text())
    review_manifest = json.loads((review / 'manifest.json').read_text())
    post_manifest = json.loads((post / 'manifest.json').read_text())
    verified(data / 'manifest.json', review_manifest['dataset_manifest_sha256'])
    population = [json.loads(x) for x in verified(data / 'synthetic-queries.jsonl', dataset_manifest['outputs_sha256']['synthetic-queries.jsonl']).splitlines()]
    mapping = json.loads((review / 'private-mapping.json').read_bytes())
    verified(review / 'blind-review.jsonl', review_manifest['output_sha256'])
    if 'private_mapping_sha256' in review_manifest:
        verified(review / 'private-mapping.json', review_manifest['private_mapping_sha256'])
    rng = random.Random(review_manifest['seed']); selected = []
    for spec in review_manifest['strata']:
        choices = sorted((r for r in population if r['intent_type'] == spec['intent_type'] and r['status'] == spec['status']), key=lambda r: r['id'])
        if len(choices) != spec['population']:
            raise ValueError('Stratum differs from frozen sample')
        selected.extend(rng.sample(choices, spec['sample']))
    rng.shuffle(selected)
    expected = [{'case_id': f'case-{i:03d}', 'query_id': r['id'], 'parent_id': r['parent_id'], 'status': r['status']} for i, r in enumerate(selected, 1)]
    if mapping != expected:
        raise ValueError('Sample mapping differs from deterministic reconstruction')
    records = [json.loads(x) for x in verified(post / 'pairs.jsonl', post_manifest['output_sha256']).splitlines()]
    if len(records) != post_manifest['generated_pairs'] or len(population) != len(records):
        raise ValueError('Dataset and postcheck sizes differ')
    post_by_key = {(r['parent_id'], r['intent_type']): r for r in records}
    if len(post_by_key) != len(records):
        raise ValueError('Duplicate postchecks')
    sample = {r['query_id'] for r in mapping}
    if len(sample) != len(mapping) or len(sample) != review_manifest['cases']:
        raise ValueError('Duplicate or incomplete sample')
    strata = defaultdict(Counter)
    found = set()
    for row in population:
        key = (row['parent_id'], row['intent_type'])
        check = post_by_key[key]
        if check['status'] != row['status']:
            raise ValueError('Dataset and postcheck status differ')
        no_anchors = not check['numeric_postcheck']['source_anchors'] and not check['numeric_postcheck']['target_anchors']
        counts = strata[row['intent_type'], row['status']]
        counts['population'] += 1; counts['population_no_anchors'] += no_anchors
        if row['id'] in sample:
            found.add(row['id']); counts['sample'] += 1; counts['sample_no_anchors'] += no_anchors
    if found != sample:
        raise ValueError('Unknown sampled query')
    return {'strata': [dict(intent_type=k[0], status=k[1], **v) for k, v in sorted(strata.items())],
            'sample_cases': len(found), 'sample_no_anchors': sum(v['sample_no_anchors'] for v in strata.values()),
            'scope': 'Anchor detection coverage, not human judgment or accuracy. Sampling remains stratified by intent and prior status, not numeric presence; subgroup sample sizes can be small. Local review texts remain private.'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
