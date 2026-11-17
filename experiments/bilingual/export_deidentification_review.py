"""Prepare a local blind paired review of English de-identification candidates."""
import hashlib
import json
import random
from pathlib import Path


def main():
    data=Path('artifacts/dataset-release/english-repair-construction-v2');repairs=Path('artifacts/english-identifier-repair-v1')
    manifest=json.loads((data/'manifest.json').read_text());payload=(data/'queries.jsonl').read_bytes();protocol=json.loads((repairs/'protocol.json').read_text())
    if hashlib.sha256(payload).hexdigest()!=manifest['output_sha256'] or protocol['dataset_sha256']!=manifest['output_sha256']:raise ValueError('Changed input')
    rows={r['id']:r for r in map(json.loads,payload.splitlines())};rng=random.Random(20261129);forms=[];mapping=[];hashes={}
    for i,rid in enumerate(protocol['ids'],1):
        path=repairs/'records'/(hashlib.sha256(rid.encode()).hexdigest()+'.json');raw=path.read_bytes();hashes[rid]=hashlib.sha256(raw).hexdigest();candidate=json.loads(raw)
        if candidate['id']!=rid or candidate['prior_status']!=rows[rid]['status']:raise ValueError('Changed candidate identity')
        if candidate.get('english') is None:continue
        variants=[('original',rows[rid]['english']),('candidate',candidate['english'])];rng.shuffle(variants)
        case=f'case-{i:03d}'
        forms.append(dict(case_id=case,version_a=variants[0][1],version_b=variants[1][1],
                          specific_identity_a=None,specific_identity_b=None,technical_equivalence=None,notes=None,reviewer_id=None))
        mapping.append(dict(case_id=case,id=rid,version_a=variants[0][0],version_b=variants[1][0],prior_status=rows[rid]['status'],candidate_status=candidate['status']))
    out=Path('artifacts/deidentification-human-review-v1');out.mkdir(parents=True,exist_ok=False)
    form=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (out/'blind-review.jsonl').write_bytes(form);(out/'private-mapping.json').write_bytes(private)
    (out/'README.txt').write_text('Review each version independently for specific person or organization identity and contact details: yes, no, or uncertain. Compare whether all technical requirements and constraints are equivalent: yes, no, or uncertain. Ignore differences solely in identifiable names. Add notes and reviewer ID. This does not by itself determine confidentiality, licensing, indirect identifiability or release rights. Do not open private-mapping.json during review.\n')
    report=dict(cases=len(forms),source_sha256=manifest['output_sha256'],candidate_protocol_sha256=hashlib.sha256((repairs/'protocol.json').read_bytes()).hexdigest(),
                candidate_record_sha256=hashes,review_sha256=hashlib.sha256(form).hexdigest(),mapping_sha256=hashlib.sha256(private).hexdigest(),seed=20261129,
                scope='Local private paired review. Candidate/original order is hidden; source text may itself identify an entity. No human judgments and no release clearance.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='candidate_record_sha256'},indent=2))


if __name__=='__main__':main()
