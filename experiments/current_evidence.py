"""Export a compact public-safe evidence inventory from completed summaries."""
import hashlib
import json
from pathlib import Path


def run():
    runs={}
    paths={
        'decisions_nips':'artifacts/public/cof/NIPS/decisions-support-v1/summary.json',
        'demand_facets':'artifacts/demand-first-facets-v2/summary.json',
        'demand_pool':'artifacts/demand-first-pool-v2/summary.json',
        'atomic_decisions':'artifacts/atomic-demand-judgments-v2/summary.json',
        'atomic_quote_audit':'artifacts/supported-atom-audit-v1/summary.json'}
    allowed=('dataset','planned','valid','plans','valid_plans','queries','mean_request_seconds','demands','structurally_valid','role_counts','jobs','judged_research_atoms','supported_atoms','partially_supported_plans','all_research_atoms_supported_plans','tasks','scope_rejected','scope_preserved_and_supported','accepted_dataset_labels','metrics')
    for name,path in paths.items():
        p=Path(path);record=json.loads(p.read_text())
        runs[name]=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),results={k:record[k] for k in allowed if k in record})
    for dataset in ('NIPS','KDD','SciRepEval'):
        p=Path('artifacts')/('array-bm25-profiles-'+dataset.lower()+'.json');r=json.loads(p.read_text())
        runs['array_bm25_'+dataset]=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),results={k:r[k] for k in ('queries','mean_seconds','array_bytes')})
        for method in ('max-term-v1','rank-fusion-v1'):
            p=Path('artifacts/public/cof')/dataset/method/'summary.json'
            if p.exists():
                r=json.loads(p.read_text());runs[method+'_'+dataset]=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),results={k:r[k] for k in ('queries','metrics')})
    report=dict(version=1,runs=runs,completion=dict(method_quality_advantage=False,high_quality_industrial_dataset=False,metric_validity=False,public_release=False),
        scope='Completed exploratory results only. Timing boundaries differ across runs. Single-model labels are not human gold. No raw demands, expert identities, quotes or model traces included.')
    Path('docs/exploration/current-evidence.json').write_text(json.dumps(report,indent=2));print('Exported',len(runs),'completed summaries')


if __name__=='__main__':run()
