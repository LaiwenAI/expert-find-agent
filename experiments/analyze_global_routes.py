"""Known-positive global routing retention; no precision or complete qrels."""
import hashlib
import json
from pathlib import Path


def run():
    root=Path('artifacts/global-expert-root-benchmark-v1');summary=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text());records=[json.loads((root/(r['id']+'.json')).read_text()) for r in summary['rows']]
    parents=sorted({r['parent'] for r in records});rows=[]
    for parent in parents:
        selected=[r for r in records if r['parent']==parent];full=next(r for r in selected if r['kind']=='full_demand');condition=next(r for r in selected if r['kind']=='silver_condition_diagnostic');requirements=[r for r in selected if r['kind']=='requirement']
        for k in protocol['budgets']:
            union=set().union(*(set(r['ranking'][:k]) for r in requirements))
            retained=any(r['known_expert_rank']<=k for r in requirements)
            rows.append(dict(parent=parent,per_requirement_budget=k,requirements=len(requirements),union_candidates=len(union),
                multipath_known_retained=retained,whole_known_retained_at_same_size=full['known_expert_rank']<=len(union),
                whole_demand_known_rank=full['known_expert_rank'],best_requirement_known_rank=min(r['known_expert_rank'] for r in requirements),
                silver_condition_known_rank=condition['known_expert_rank'],available_pool_fraction=len(union)/summary['experts']))
    report=dict(version=1,indexed_experts=summary['experts'],unavailable_experts=summary['unavailable_experts'],known_pairs=len(parents),rows=rows,
        mean_query_encoding_seconds=sum(r['query_encoding_seconds'] for r in records)/len(records),mean_retrieval_seconds=sum(r['retrieval_seconds'] for r in records)/len(records),
        root_encoding_seconds=summary['root_encoding_seconds'],summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        scope='Known silver positives only, repeated development cases. Condition query is a diagnostic informed by silver construction, not an original-demand deployment result. No complete recall/precision or safe semantic-pruning guarantee. Missing-data experts remain a separate unknown bucket. Planning and final evidence verification excluded from timing.')
    Path('docs/exploration/global-root-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
