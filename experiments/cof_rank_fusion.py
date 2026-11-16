"""Fixed reciprocal-rank fusion of complementary classical expert rankers."""
import json
from pathlib import Path
from urllib.parse import quote
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load,rank_judged,precision_metrics
from expert_search.packing_analysis import paired_interval


def run(dataset):
    root,papers,reviewers,labels=load(dataset)
    out=root/'rank-fusion-v1';out.mkdir(exist_ok=True)
    protocol=dict(version=1,dataset=dataset,methods=['model1','bm25-top3'],constant=60,weights=[1,1],
        tuning='None; fixed exploratory combination after inspecting baseline dataset means. Not a held-out confirmation.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    engine=ArrayBM25(root/'unique-profile-corpus/fast.sqlite');profiles={eid:r['papers'] for eid,r in reviewers.items()}
    rows=[]
    for qid in sorted(labels['soft']):
        lm=json.loads((root/'scale-check-v1'/(quote(qid,safe='')+'.json')).read_text())
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        scores=engine.profile_scores(query,profiles)
        lexical=rank_judged({eid:s['bm25-top3'] for eid,s in scores.items()},labels['soft'][qid])
        if set(lm['ranking'])!=set(lexical):raise ValueError('Different candidate sets')
        contributions={eid:0. for eid in lexical}
        for ranking in (lm['ranking'],lexical):
            for rank,eid in enumerate(ranking,1):contributions[eid]+=1/(60+rank)
        fused=sorted(contributions,key=lambda eid:(-contributions[eid],eid))
        metrics={name:{task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()} for name,ranking in [('model1',lm['ranking']),('bm25-top3',lexical),('fusion',fused)]}
        row=dict(query_id=qid,ranking=fused,metrics=metrics);rows.append(row)
        (out/(quote(qid,safe='')+'.json')).write_text(json.dumps(row,indent=2))
    report=dict(protocol=protocol,queries=len(rows),metrics={name:{task:{metric:sum(r['metrics'][name][task][metric] for r in rows)/len(rows) for metric in ('p@5','ndcg@10')} for task in labels} for name in ('model1','bm25-top3','fusion')},
        paired_differences={name:{task:{metric:paired_interval([r['metrics']['fusion'][task][metric]-r['metrics'][name][task][metric] for r in rows]) for metric in ('p@5','ndcg@10')} for task in labels} for name in ('model1','bm25-top3')},
        timing='Cached LM rankings; no online fusion latency claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='paired_differences'},indent=2),flush=True)


if __name__=='__main__':
    for dataset in ('KDD','SciRepEval'):run(dataset)
