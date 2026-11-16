"""Static-size and candidate-pool diagnostics; no label-based method selection."""
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.query_lift import query_lift


def run(dataset):
    _,_,reviewers,labels=load(dataset);root=Path('artifacts')/('cof-source-fusion-'+dataset.lower()+'-v1')
    sizes={e:len(set(r['papers'])) for e,r in reviewers.items()};rows=[]
    for qid,truth in sorted(labels['soft'].items()):
        saved=json.loads((root/(quote(qid,safe='')+'.json')).read_text())
        prior=sorted(truth,key=lambda e:(-min(3,sizes[e]),e))
        metrics={t:precision_metrics(prior,q[qid]) for t,q in labels.items()}
        rows.append(dict(query_id=qid,candidates=len(truth),rank_sensitive_p5=len(truth)>5,size_prior_metrics=metrics,
                         delta_soft_p5=saved['metrics']['source_top3']['soft']['p@5']-saved['metrics']['qwen_max']['soft']['p@5'],
                         positive_sparse_profiles=sum(v>0 and sizes[e]<3 for e,v in truth.items()),positive_profiles=sum(v>0 for v in truth.values())))
    report=dict(dataset=dataset,queries=len(rows),rank_sensitive_p5_queries=sum(r['rank_sensitive_p5'] for r in rows),
                profile_counts=dict(clipped_size_score_levels=len({min(3,n) for n in sizes.values()}),total=len(sizes),fewer_than_three=sum(n<3 for n in sizes.values()),median=float(np.median(list(sizes.values())))),
                size_prior_p5={t:float(np.mean([r['size_prior_metrics'][t]['p@5'] for r in rows])) for t in labels},
                delta_soft_p5_by_pool={name:dict(queries=len(group),mean=float(np.mean([r['delta_soft_p5'] for r in group])) if group else None) for name,group in [('at_most_5',[r for r in rows if not r['rank_sensitive_p5']]),('more_than_5',[r for r in rows if r['rank_sensitive_p5']])]},rows=rows,
                warning='Exploratory bias diagnostics, not causal attribution or adjusted significance. Size prior is query independent except the supplied eligible pool. Repeated expert/label occurrences are not independent samples; no strategy selected from these outcomes.')
    if dataset=='SIGIR':
        saved={qid:json.loads((root/(quote(qid,safe='')+'.json')).read_text()) for qid in labels['soft']}
        runs={m:{qid:r['rankings'][m] for qid,r in saved.items()} for m in next(iter(saved.values()))['rankings']}
        runs['clipped_size_prior']={qid:sorted(truth,key=lambda e:(-min(3,sizes[e]),e)) for qid,truth in labels['soft'].items()}
        report['query_association']={m:{t:query_lift(run,truth) for t,truth in labels.items()} for m,run in runs.items()}
    else:report['query_association']='Not applied: query-dependent sparse eligibility invalidates a common-universe shuffle comparison.'
    (root/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    for dataset in ('SIGIR','NIPS'):run(dataset)
