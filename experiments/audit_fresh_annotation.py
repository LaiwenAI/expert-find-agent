"""Diagnose frozen annotation coverage without imputing or changing labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def run():
    root=Path('artifacts/fresh-method-data-v1/annotation-development-v1')
    files=sorted(root.glob('demand-*.json'));statuses=Counter();errors=Counter();reasons=Counter();confusion=Counter();audits=Counter();inputs={}
    for path in files:
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();row=json.loads(path.read_text())
        if row['split']!='development':raise ValueError('Heldout annotation encountered')
        passes=[]
        for p in row['passes']:
            statuses[p['status']]+=1
            if p['status']!='structurally_valid':errors[p.get('error_type','unknown')]+=1
            passes.append({r['expert_id']:r['level'] for r in p.get('result',{}).get('ranking',[])} if p['status']=='structurally_valid' else {})
        for judgment in row['judgments']:
            reasons[judgment['reason']]+=1;eid=judgment['expert_id']
            if eid in passes[0] and eid in passes[1]:confusion[(passes[0][eid],passes[1][eid])]+=1
        audits.update(row.get('witness_audit',{}).get('labels',{}).values())
    report=dict(queries=len(files),pass_status=dict(statuses),pass_errors=dict(errors),admission_reasons=dict(reasons),
                paired_valid_candidate_labels=sum(confusion.values()),paired_label_agreements=sum(v for (a,b),v in confusion.items() if a==b),
                paired_confusion=[dict(first=a,second=b,count=v) for (a,b),v in sorted(confusion.items())],witness_audit_choices=dict(audits),
                inputs_sha256=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Coverage/error audit only. Pair agreement and witness decisions are same-model observations, not semantic accuracy. No retries, new labels or heldout calls.')
    out=Path('artifacts/fresh-annotation-audit-v1');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
