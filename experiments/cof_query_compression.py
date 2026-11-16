"""Frozen top16 IDF/within-query-frequency lexical compression control."""
from collections import Counter
import json
import math
import random
import time
from pathlib import Path
from urllib.parse import quote
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load,rank_judged,precision_metrics
from expert_search.retrieval import tokens
from expert_search.packing_analysis import paired_interval


def run(dataset):
    root,papers,reviewers,labels=load(dataset);out=root/'query-compression-v1';out.mkdir(exist_ok=True)
    protocol=dict(version=1,dataset=dataset,terms=16,rule='Select in-vocabulary original unique tokens by IDF * (1+log query term frequency), descending; lexical tie break. No expansion or training. BM25-top3 aggregation unchanged.',tuning='Fixed before this run; exploratory reused datasets, no heldout claim.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed compression protocol')
    marker.write_text(json.dumps(protocol,indent=2));engine=ArrayBM25(root/'unique-profile-corpus/fast.sqlite');profiles={eid:r['papers'] for eid,r in reviewers.items()};n=len(engine.ids);rows=[]
    for qid in sorted(labels['soft']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        order=['full','compressed'];random.Random(qid).shuffle(order);results={}
        for mode in order:
            start=time.perf_counter();selected=None
            if mode=='compressed':
                counts=Counter(tokens(query));weights={t:math.log(1+(n-len(engine.postings[t][0])+.5)/(len(engine.postings[t][0])+.5))*(1+math.log(c)) for t,c in counts.items() if t in engine.postings}
                selected=sorted(weights,key=lambda t:(-weights[t],t))[:16];current=' '.join(selected)
            else:current=query
            scores=engine.profile_scores(current,profiles)
            ranking=rank_judged({eid:s['bm25-top3'] for eid,s in scores.items()},labels['soft'][qid]);seconds=time.perf_counter()-start
            results[mode]=dict(ranking=ranking,seconds=seconds,selected_terms=selected,metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()})
        row=dict(query_id=qid,results=results);rows.append(row);(out/(quote(qid,safe='')+'.json')).write_text(json.dumps(row,indent=2))
    report=dict(protocol=protocol,queries=len(rows),mean_seconds={mode:sum(r['results'][mode]['seconds'] for r in rows)/len(rows) for mode in ('full','compressed')},
        metrics={mode:{task:{metric:sum(r['results'][mode]['metrics'][task][metric] for r in rows)/len(rows) for metric in ('p@5','ndcg@10')} for task in labels} for mode in ('full','compressed')},
        paired_differences={task:{metric:paired_interval([r['results']['compressed']['metrics'][task][metric]-r['results']['full']['metrics'][task][metric] for r in rows]) for metric in ('p@5','ndcg@10')} for task in labels})
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='paired_differences'},indent=2),flush=True)


if __name__=='__main__':
    for dataset in ('NIPS','KDD','SciRepEval'):run(dataset)
