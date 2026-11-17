"""Export unjudged real cases for independent, model-blind local review."""
import hashlib
import json
import random
from pathlib import Path


def main():
    source = Path('artifacts/bilingual-order-controls-v1/cases.jsonl')
    records = [json.loads(line) for line in source.read_text().splitlines()]
    cases = [r for r in records if r['group'] == 'real']
    if len(cases) != 30 or len({r['id'] for r in cases}) != 30:
        raise ValueError('Expected 30 distinct real cases')
    random.Random(20261117).shuffle(cases)
    root = Path('artifacts/grounding-human-review-v1')
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        p = case['payload']
        rows.append(dict(case_id=case['id'], intent_type=p['intent_type'],
                         original_need=p['original_need'], cited_segments=p['cited_segments'],
                         chinese_request=p['zh'], english_request=p['en'],
                         chinese_grounding=None, english_grounding=None,
                         bilingual_equivalence=None, rationale=None, reviewer_id=None))
    output = root / 'blind-review.jsonl'
    if output.exists():
        raise FileExistsError('Preserve existing review; use a new version for a new export')
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    output.write_bytes(payload)
    manifest = dict(cases=len(rows), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    output_sha256=hashlib.sha256(payload).hexdigest(), shuffle_seed=20261117,
                    allowed_labels=['supported', 'changed_or_unsupported', 'uncertain'],
                    scope='Local sensitive text; independent review required; no model predictions included.')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
