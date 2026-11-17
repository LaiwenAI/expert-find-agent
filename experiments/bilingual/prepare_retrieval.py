"""Freeze one derived intent per real parent for bilingual retrieval diagnostics."""
import hashlib
import json
from pathlib import Path


def prepare():
    dataset = Path('artifacts/dataset-release/bilingual-query-construction-v1')
    source = dataset / 'synthetic-queries.jsonl'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['outputs_sha256'][source.name]:
        raise ValueError('Changed bilingual construction dataset')
    rows = list(map(json.loads, source.read_text().splitlines()))
    kinds = list(manifest['taxonomy'])
    by_parent = {}
    for row in rows:
        by_parent.setdefault(row['parent_id'], {})[row['intent_type']] = row
    selected = []
    for index, qid in enumerate(sorted(by_parent)):
        row = by_parent[qid][kinds[index % len(kinds)]]
        if row['split'] != 'construction' or not row['zh'] or not row['en']:
            raise ValueError('Invalid paired construction query')
        selected.append({k: row[k] for k in ('id', 'parent_id', 'group_id', 'intent_type', 'zh', 'en', 'status')})
    if len(selected) != 1000 or len({r['parent_id'] for r in selected}) != 1000:
        raise ValueError('Need 1000 unique real-parent-derived pairs')
    out = Path('artifacts/bilingual-retrieval-v1')
    out.mkdir(exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
    path = out / 'queries.jsonl'
    if path.exists() and path.read_text() != body:
        raise ValueError('Changed frozen query selection')
    path.write_text(body)
    plan = dict(derived_query_pairs=len(selected), independent_real_parents=1000, language_views=2000, taxonomy_counts={k: sum(r['intent_type'] == k for r in selected) for k in kinds}, status_counts={s: sum(r['status'] == s for r in selected) for s in sorted({r['status'] for r in selected})}, input_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest(), str(dataset / 'manifest.json'): hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()}, queries_sha256=hashlib.sha256(body.encode()).hexdigest(), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Deterministic one-intent-per-parent construction diagnostic; all statuses retained, each type200. Paired languages do not double independent query count. No expert relevance labels or full-original English translations. Measure frozen-score fidelity and language ranking stability, not correctness.')
    marker = out / 'selection.json'
    if marker.exists() and json.loads(marker.read_text()) != plan:
        raise ValueError('Changed retrieval selection protocol')
    marker.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    prepare()
