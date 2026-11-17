"""Export a targeted blind review supplement for an uncovered numeric subgroup."""
import hashlib
import json
import random
from pathlib import Path
from .audit_review_numeric_coverage import run, verified


def main():
    coverage = run()
    data = Path('artifacts/dataset-release/bilingual-query-construction-v1')
    review = Path('artifacts/stratified-grounding-human-review-v1')
    post = Path('artifacts/large-bilingual-data-v1/postvalidation-v2')
    manifest = json.loads((data / 'manifest.json').read_text())
    post_manifest = json.loads((post / 'manifest.json').read_text())
    parents = {r['id']: r for r in map(json.loads, verified(data / 'real-queries.jsonl', manifest['outputs_sha256']['real-queries.jsonl']).splitlines())}
    population = [json.loads(x) for x in verified(data / 'synthetic-queries.jsonl', manifest['outputs_sha256']['synthetic-queries.jsonl']).splitlines()]
    checks = {(r['parent_id'], r['intent_type']): r for r in map(json.loads, verified(post / 'pairs.jsonl', post_manifest['output_sha256']).splitlines())}
    previously_sampled = {r['query_id'] for r in json.loads((review / 'private-mapping.json').read_text())}
    spec = next(r for r in coverage['strata'] if r['intent_type'] == 'evidence_lookup' and r['status'] == 'unjudged_pair')
    if spec['sample_no_anchors'] != 0:
        raise ValueError('Target subgroup is already represented; revise supplemental protocol')
    eligible = []
    for row in population:
        check = checks[row['parent_id'], row['intent_type']]['numeric_postcheck']
        if row['intent_type'] == 'evidence_lookup' and row['status'] == 'unjudged_pair' and not check['source_anchors'] and not check['target_anchors'] and row['id'] not in previously_sampled:
            eligible.append(row)
    eligible.sort(key=lambda r: r['id'])
    if len(eligible) != spec['population_no_anchors'] or len(eligible) < 5:
        raise ValueError('Unexpected eligible population')
    sample = random.Random(20261121).sample(eligible, 5)
    forms = []; mapping = []
    for i, row in enumerate(sample, 1):
        case = f'supplement-{i:03d}'
        forms.append(dict(case_id=case, intent_type=row['intent_type'], original_need=parents[row['parent_id']]['zh'],
                          chinese_request=row['zh'], english_request=row['en'], chinese_grounding=None,
                          english_grounding=None, bilingual_equivalence=None, rationale=None, reviewer_id=None))
        mapping.append(dict(case_id=case, query_id=row['id'], parent_id=row['parent_id']))
    root = Path('artifacts/numeric-gap-grounding-human-review-v1')
    root.mkdir(parents=True, exist_ok=False)
    form_bytes = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in forms).encode()
    map_bytes = (json.dumps(mapping, indent=2) + '\n').encode()
    (root / 'blind-review.jsonl').write_bytes(form_bytes)
    (root / 'private-mapping.json').write_bytes(map_bytes)
    report = dict(cases=len(forms), eligible=len(eligible), seed=20261121,
                  dataset_manifest_sha256=hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
                  postcheck_sha256=post_manifest['output_sha256'], blind_review_sha256=hashlib.sha256(form_bytes).hexdigest(),
                  private_mapping_sha256=hashlib.sha256(map_bytes).hexdigest(),
                  scope='Targeted supplement from an uncovered subgroup; not part of the original stratified sample. Sensitive texts remain local. No human judgments; do not pool without a revised sampling design.')
    (root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
