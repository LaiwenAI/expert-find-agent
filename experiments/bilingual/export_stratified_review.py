"""Prepare a screen-blind, stratified local query-grounding review packet."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    data=Path('artifacts/dataset-release/bilingual-query-construction-v1')
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:raise ValueError('Dataset hash mismatch')
    parents={r['id']:r for r in map(json.loads,(data/'real-queries.jsonl').read_text().splitlines())}
    strata=defaultdict(list)
    for row in map(json.loads,(data/'synthetic-queries.jsonl').read_text().splitlines()):
        if row['parent_id'] not in parents or row['split']!='construction':raise ValueError('Invalid parent or split')
        strata[row['intent_type'],row['status']].append(row)
    statuses={'retained_model_screened_pair','unjudged_pair'}
    if len(strata)!=10 or {s for _,s in strata}!=statuses:raise ValueError('Unexpected strata')
    rng=random.Random(20261120);selected=[];sizes=[]
    for (kind,status),rows in sorted(strata.items()):
        rows.sort(key=lambda r:r['id'])
        if len(rows)<10 or len({r['id'] for r in rows})!=len(rows):raise ValueError('Invalid stratum')
        sizes.append({'intent_type':kind,'status':status,'population':len(rows),'sample':10})
        selected.extend(rng.sample(rows,10))
    rng.shuffle(selected);review=[];mapping=[]
    for i,row in enumerate(selected,1):
        case_id=f'case-{i:03d}'
        review.append(dict(case_id=case_id,intent_type=row['intent_type'],original_need=parents[row['parent_id']]['zh'],
                           chinese_request=row['zh'],english_request=row['en'],chinese_grounding=None,
                           english_grounding=None,bilingual_equivalence=None,rationale=None,reviewer_id=None))
        mapping.append(dict(case_id=case_id,query_id=row['id'],parent_id=row['parent_id'],status=row['status']))
    root=Path('artifacts/stratified-grounding-human-review-v1');root.mkdir(parents=True,exist_ok=False)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review).encode()
    (root/'blind-review.jsonl').write_bytes(payload)
    mapping_payload=(json.dumps(mapping,indent=2)+'\n').encode()
    (root/'private-mapping.json').write_bytes(mapping_payload)
    report=dict(cases=len(review),seed=20261120,strata=sizes,output_sha256=hashlib.sha256(payload).hexdigest(),
                dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
                private_mapping_sha256=hashlib.sha256(mapping_payload).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Local sensitive review packet, not public. Sampled uniformly within type/status strata; status and parent IDs are hidden from review forms. No completed human judgments. Parents can recur; use parent clusters for uncertainty and stratum weights for population estimates.')
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
