"""Create a local candidate view with conservative bilingual recheck status."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .audit_release_identifiers import PATTERNS

DATA=Path('artifacts/dataset-release/english-deidentification-candidates-v1')
AUDIT=Path('artifacts/deidentified-alignment-audit-v1')
OUT=Path('artifacts/dataset-release/english-alignment-gated-v1')


def apply_alignment(row, audit):
    if row['id']!=audit['id'] or row.get('deidentification')!='same_model_screened_candidate':raise ValueError('Audit identity mismatch')
    arms=audit.get('arms',{})
    if set(arms)!={'old','new'}:raise ValueError('Incomplete audit')
    passed=all(arms[a].get('status')=='valid' and arms[a].get('label')=='equivalent' for a in ('old','new'))
    prior=row['status']
    if prior not in ('retained_model_screened_pair','unjudged_pair'):raise ValueError('Unknown source status')
    return dict(row,status=prior if passed else 'unjudged_pair',alignment_recheck='both_equivalent' if passed else 'unjudged')


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());raw=(DATA/'queries.jsonl').read_bytes();protocol=json.loads((AUDIT/'protocol.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=manifest['output_sha256']:raise ValueError('Changed data')
    if (manifest['source_sha256']!=protocol['english_sha256'] or len(protocol['candidate_sha256'])!=18 or
            not set(protocol['candidate_sha256']).issubset(manifest['candidate_record_sha256'])):
        raise ValueError('Changed candidate cohort or English source')
    source=Path('artifacts/dataset-release/bilingual-query-construction-v1/synthetic-queries.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=protocol['source_sha256']:
        raise ValueError('Changed Chinese source')
    for key,digest in protocol['candidate_sha256'].items():
        if manifest['candidate_record_sha256'].get(key)!=digest:raise ValueError('Changed candidate record')
    rows=[json.loads(s) for s in raw.splitlines()];result=[];audit_hashes={}
    if len(rows)!=5000 or len({r['id'] for r in rows})!=5000:raise ValueError('Invalid view')
    for row in rows:
        if row['id'] in protocol['candidate_sha256']:
            path=AUDIT/'records'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json');body=path.read_bytes();audit_hashes[row['id']]=hashlib.sha256(body).hexdigest()
            result.append(apply_alignment(row,json.loads(body)))
        else:result.append(dict(row,alignment_recheck='not_assessed'))
    if set(audit_hashes)!=set(protocol['candidate_sha256']):raise ValueError('Missing audit records')
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result).encode()
    if PATTERNS['han_or_radical'].search(payload.decode()):raise ValueError('Han character remains')
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'queries.jsonl').write_bytes(payload)
    report=dict(rows=len(result),statuses=dict(Counter(r['status'] for r in result)),alignment_recheck=dict(Counter(r['alignment_recheck'] for r in result)),
                prior_view_sha256=manifest['output_sha256'],audit_protocol_sha256=hashlib.sha256((AUDIT/'protocol.json').read_bytes()).hexdigest(),
                audit_record_sha256=audit_hashes,output_sha256=hashlib.sha256(payload).hexdigest(),
                scope='Local candidate view; same-model bilingual disagreement only downgrades status. No promotion, human gold, identity assurance or release clearance.')
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='audit_record_sha256'},indent=2))


if __name__=='__main__':main()
