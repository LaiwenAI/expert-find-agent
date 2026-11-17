"""Snapshot numeric rechecks without modifying paid generation/audit records."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from expert_search.bilingual.postvalidation import check_pair


def run():
    root = Path('artifacts/large-bilingual-data-v1/synthesis-v1')
    protocol = root / 'protocol.json'
    selected = json.loads(protocol.read_text())['selected_parent_ids']
    output = Path('artifacts/large-bilingual-data-v1/postvalidation-v2')
    output.mkdir(exist_ok=True)
    rows, hashes = [], {}
    counts = dict(match=0, mismatch=0, uncertain=0, previous_screened=0, retained_screened=0, downgraded=0)
    for qid in selected:
        path = root / 'parents' / (qid + '.json')
        if not path.exists():
            continue
        body = path.read_bytes()
        value = json.loads(body)
        if value['parent_id'] != qid or value['status'] not in ('unresolved', 'completed_with_unjudged_retained'):
            raise ValueError('Invalid parent terminal record')
        hashes[qid] = hashlib.sha256(body).hexdigest()
        for kind, variant in value['variants'].items():
            check = check_pair(variant['zh'], variant['en'])
            previous = variant.get('label_status') == 'model_screened_query_pair'
            retained = previous and check['status'] == 'match'
            counts[check['status']] += 1
            counts['previous_screened'] += previous
            counts['retained_screened'] += retained
            counts['downgraded'] += previous and not retained
            rows.append(dict(parent_id=qid, intent_type=kind, prior_status=variant.get('label_status'), numeric_postcheck=check, status='retained_model_screened_pair' if retained else 'unjudged_pair', expert_fit_grade=None))
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    (output / 'pairs.jsonl').write_text(payload)
    manifest = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), terminal_parents=len(hashes), generated_pairs=len(rows), counts=counts,
                    inputs_sha256=hashes, protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                    code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('src/expert_search/bilingual/postvalidation.py'), Path('src/expert_search/bilingual/constraints.py')]},
                    output_sha256=hashlib.sha256(payload.encode()).hexdigest(), scope='Snapshot only. Conservative intersection with previous screen; no promotions, no regeneration, no independent semantic or expert-fit labels. Original terminal records unchanged. Unsupported negation is unresolved; numeric-free pairs have no numeric equivalence guarantee.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('inputs_sha256', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    run()
