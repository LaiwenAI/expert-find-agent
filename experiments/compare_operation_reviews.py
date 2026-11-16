"""Compare all paired operation reviews without turning agreement into gold."""
import hashlib
import json
from pathlib import Path


def run():
    roots=[Path('artifacts/demand-operation-audit-v1'),Path('artifacts/demand-operation-audit-luna-v1')]
    plan_root=Path('artifacts/demand-operation-plans-v1')
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text())
        for name,digest in protocol['parents'].items():
            if hashlib.sha256((plan_root/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed audited demand plan')
    keys=json.loads((plan_root/'protocol.json').read_text())['queries']; rows=[]
    for key in keys:
        plan=json.loads((plan_root/(key+'.json')).read_text())
        if plan['status']!='structurally_valid':
            rows.append(dict(id=key,status='planning_unresolved'));continue
        files=[root/(key+'.json') for root in roots]
        if any(not f.exists() for f in files):raise ValueError('Paired review still incomplete')
        reviews=[json.loads(f.read_text()) for f in files]
        if any(r['status']!='valid' for r in reviews):
            rows.append(dict(id=key,status='review_unresolved'));continue
        indexed=[{o['id']:o for o in r['audit']['operations']} for r in reviews]
        ids=[o['id'] for o in plan['plan']['operations']]
        matches={k:sum(indexed[0][i][k]==indexed[1][i][k] for i in ids) for k in ('anchored','atomic','scope_preserved')}
        rows.append(dict(id=key,status='paired',operations=len(ids),agreement=matches,
                         both_pass=sum(all(index[i][k] for index in indexed for k in matches) for i in ids),
                         duplicate_groups=[r['audit']['duplicate_groups'] for r in reviews],
                         missing_counts=[len(r['audit']['missing']) for r in reviews],
                         obligations_preserved=[r['audit']['obligations_preserved'] for r in reviews]))
    paired=[r for r in rows if r['status']=='paired']
    report=dict(plans=len(rows),paired_plans=len(paired),operations=sum(r['operations'] for r in paired),
                agreement={k:sum(r['agreement'][k] for r in paired) for k in ('anchored','atomic','scope_preserved')},
                both_pass=sum(r['both_pass'] for r in paired),rows=rows,
                warning='Agreement is not correctness. Both reviewers share a provider; neither labels nor operation coverage are human validated. Disagreements and potentially false missing reports retained; no automatic selection or deletion.')
    out=Path('artifacts/demand-operation-review-comparison-v1');out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
