"""Prepare a local human review of the unflagged stratified sample."""
import hashlib
import json
import random
from pathlib import Path


def main():
    data=Path('artifacts/dataset-release/english-alignment-gated-v1');audit=Path('artifacts/unflagged-identity-sample-v1')
    manifest=json.loads((data/'manifest.json').read_text());raw=(data/'queries.jsonl').read_bytes();protocol=json.loads((audit/'protocol.json').read_text())
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=manifest['output_sha256'] or digest!=protocol['source_sha256']:raise ValueError('Changed source')
    rows={r['id']:r for r in map(json.loads,raw.splitlines())};selected=protocol['selected_ids']
    if len(selected)!=200 or len(set(selected))!=200:raise ValueError('Changed sample')
    items=[];hashes={}
    for rid in selected:
        path=audit/'records'/(hashlib.sha256(rid.encode()).hexdigest()+'.json');body=path.read_bytes();hashes[rid]=hashlib.sha256(body).hexdigest();result=json.loads(body);row=rows[rid]
        if result['id']!=rid or row['deidentification']!='not_assessed' or result['intent_type']!=row['intent_type'] or result['prior_status']!=row['status']:raise ValueError('Changed sample record')
        items.append((row,result))
    random.Random(20261201).shuffle(items);forms=[];mapping=[]
    for i,(row,result) in enumerate(items,1):
        cid=f'case-{i:03d}'
        forms.append(dict(case_id=cid,english_request=row['english'],specific_identity=None,indirect_identifiers=None,notes=None,reviewer_id=None))
        mapping.append(dict(case_id=cid,id=row['id'],intent_type=row['intent_type'],status=row['status'],model_status=result['status'],model_label=result.get('label')))
    out=Path('artifacts/unflagged-identity-human-review-v1');out.mkdir(parents=True,exist_ok=False)
    form=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (out/'blind-review.jsonl').write_bytes(form);(out/'private-mapping.json').write_bytes(private)
    (out/'README.txt').write_text('Review each request for a specific person or named organization, contact detail, URL or street address: yes, no, or uncertain. Record possible indirect identifiers separately, with a short explanation. This sample and any review do not determine confidentiality or rights. Do not read private-mapping.json during review.\n')
    report=dict(cases=len(forms),source_sha256=digest,audit_protocol_sha256=hashlib.sha256((audit/'protocol.json').read_bytes()).hexdigest(),
                audit_record_sha256=hashes,review_sha256=hashlib.sha256(form).hexdigest(),mapping_sha256=hashlib.sha256(private).hexdigest(),seed=20261201,
                scope='Local sensitive English sample; model predictions, IDs and strata are hidden. Equal sample counts across ten strata require weighting for population summaries. Parent needs can recur. No human judgments and no release clearance.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='audit_record_sha256'},indent=2))


if __name__=='__main__':main()
