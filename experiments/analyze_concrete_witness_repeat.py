"""Repeated blind screens of frozen claims: reproducibility, never semantic gold."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def run():
    data=Path('artifacts/concrete-partial-comparison-v1');roots=[Path('artifacts/concrete-witness-audit-v1'),Path('artifacts/concrete-witness-repeat-v1')];protocol=json.loads((data/'protocol.json').read_text());counts={a:Counter() for a in ['focused','concrete']};inputs={};overlap=[];equal=0
    for qid in protocol['query_ids']:
        files=[r/(qid+'.json') for r in roots];rows=[json.loads(p.read_text()) for p in files]
        if rows[0]['input_sha256']!=rows[1]['input_sha256'] or rows[0]['origins']!=rows[1]['origins']:raise ValueError('Changed blind audit input/order')
        for p in files:inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for name,origin in rows[0]['origins'].items():
            labels=[r.get('audit',{}).get('labels',{}).get(name) for r in rows];c=counts[origin['arm']];c['actionable_claims']+=1
            if any(l is None for l in labels):c['missing_one_or_both_verdicts']+=1;continue
            c['paired_verdicts']+=1;c['category_agreements']+=labels[0]==labels[1]
            c['both_supported']+=all(l=='supported_scoped_claim' for l in labels)
            c['both_generic_or_overclaimed']+=all(l=='generic_or_overclaimed' for l in labels)
            c['supported_status_disagreements']+=((labels[0]=='supported_scoped_claim')!=(labels[1]=='supported_scoped_claim'))
        generated=json.loads((data/(qid+'.json')).read_text());rankings=[]
        for arm in ['focused','concrete']:
            r=generated['arms'][arm]['review']
            if r['status']!='structurally_valid':break
            rankings.append([x['expert_id'] for x in r['result']['ranking']])
        if len(rankings)==2:overlap.append(len(set(rankings[0])&set(rankings[1]))/3);equal+=rankings[0]==rankings[1]
    report=dict(queries=len(protocol['query_ids']),arms={a:dict(c) for a,c in counts.items()},paired_valid_generation_queries=len(overlap),ordered_top3_agreements=equal,mean_top3_set_overlap=sum(overlap)/len(overlap) if overlap else None,inputs_sha256=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Identical frozen proposal input and blind order, two fresh same-model native audits. Agreement/intersection supports repeatability only, not correctness, calibration or independent expert accuracy. Missing verdicts stay unresolved; no imputation. Chinese expression and bridge specificity changed together, so effects are not attributed to either factor alone.')
    output=Path('artifacts/concrete-witness-repeat-analysis-v1');output.mkdir(exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
