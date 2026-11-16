"""Fixed lexical per-term max aggregation, all queries without tuning."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote
from expert_search.max_term_profiles import MaxTermProfiles
from expert_search.cof_benchmark import load,rank_judged,precision_metrics
from expert_search.packing_analysis import paired_interval


def run(dataset):
    root,papers,reviewers,labels=load(dataset)
    out=root/'max-term-v1';out.mkdir(exist_ok=True)
    protocol=dict(version=1,dataset=dataset,rule='Sum over unique query tokens of maximum precomputed document BM25 contribution per expert, across unique owned papers.',training=False,tuning=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start=time.perf_counter();engine=MaxTermProfiles(root/'unique-profile-corpus/fast.sqlite',{eid:r['papers'] for eid,r in reviewers.items()});initialization=time.perf_counter()-start
    rows=[]
    for qid in sorted(labels['soft']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        start=time.perf_counter();scores=engine.score(query)
        ranking=rank_judged(scores,labels['soft'][qid]);seconds=time.perf_counter()-start
        reference=json.loads((root/'baseline-lexical'/(quote(qid,safe='')+'.json')).read_text())
        row=dict(query_id=qid,ranking=ranking,seconds=seconds,metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()},reference=reference['results'])
        rows.append(row);(out/(quote(qid,safe='')+'.json')).write_text(json.dumps(row,indent=2))
    report=dict(protocol=protocol,queries=len(rows),initialization_seconds=initialization,array_bytes=engine.array_bytes,mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        metrics={task:{metric:sum(r['metrics'][task][metric] for r in rows)/len(rows) for metric in ('p@5','ndcg@10')} for task in labels},
        paired_differences={method:{task:{metric:paired_interval([r['metrics'][task][metric]-r['reference'][task][method]['metrics'][metric] for r in rows]) for metric in ('p@5','ndcg@10')} for task in labels} for method in ('bm25-max','bm25-top3')},
        warning='Topic retrieval only; per-term cross-paper evidence does not prove a joint research relation. Single timing repetition, excludes index loading.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='paired_differences'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='NIPS');run(parser.parse_args().dataset)
