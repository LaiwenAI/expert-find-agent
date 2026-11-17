"""Summarize the coverage, not accuracy, of frozen numeric postchecks."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    payload = (root / 'pairs.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['output_sha256']:
        raise ValueError('Postcheck output checksum mismatch')
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) != manifest['generated_pairs'] or len({(r['parent_id'], r['intent_type']) for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate pair records')
    def aggregate(items):
        result = collections.Counter()
        for row in items:
            check = row['numeric_postcheck']; src = check['source_anchors']; dst = check['target_anchors']
            retained = row['status'] == 'retained_model_screened_pair'
            result['pairs'] += 1; result['retained'] += retained
            result['no_anchors_either_view'] += not src and not dst
            result['retained_no_anchors_either_view'] += retained and not src and not dst
            result['retained_with_anchors_both_views'] += retained and bool(src) and bool(dst)
            result['retained_with_operator_both_views'] += retained and any(a['operator'] for a in src) and any(a['operator'] for a in dst)
        return dict(result)
    overall = aggregate(rows)
    if overall['retained'] != manifest['counts']['retained_screened']:
        raise ValueError('Retained count differs from manifest')
    return {'overall': overall, 'by_intent': {kind: aggregate([r for r in rows if r['intent_type'] == kind]) for kind in sorted({r['intent_type'] for r in rows})},
            'output_sha256': manifest['output_sha256'],
            'scope': 'Coverage of detected numeric anchors only. No-anchor pairs pass the numeric guard vacuously; anchors may be missed, and matching anchors do not prove semantic equivalence or expert relevance.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/large-bilingual-data-v1/postvalidation-v2'))
    print(json.dumps(analyze(parser.parse_args().root), indent=2))
