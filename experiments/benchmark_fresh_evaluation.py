"""Within-view development baselines; incomplete silver never becomes closed-world gold."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from expert_search.retrieval import tokens
from expert_search.evaluation_gain import actionable_gain_interval,snapshot_gain_interval,condensed_ndcg


def run(directory):
    root=Path(directory);manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if name in {'queries.jsonl','pools.jsonl','snapshots.jsonl','judgments.jsonl','corpus.jsonl','dense-baseline-runs.jsonl'}:
            with (root/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Changed evaluation dataset member')
    queries={q['id']:q for q in map(json.loads,(root/'queries.jsonl').read_text().splitlines())}
    pools={p['query_id']:p for p in map(json.loads,(root/'pools.jsonl').read_text().splitlines())}
    snapshots={(r['query_id'],r['expert_id']):r for r in map(json.loads,(root/'snapshots.jsonl').read_text().splitlines())}
    judgments={}
    for row in map(json.loads,(root/'judgments.jsonl').read_text().splitlines()):judgments.setdefault(row['query_id'],{})[row['expert_id']]=row['grade']
    needed={s['source_id'] for r in snapshots.values() for s in r['sources']};tf={};lengths={};df={s:Counter() for s in ['development','heldout']};n=Counter();total=Counter()
    start=time.perf_counter()
    for doc in map(json.loads,(root/'corpus.jsonl').read_text().splitlines()):
        counts=Counter(tokens(doc['text'][:1200]));split=doc['split'];df[split].update(counts.keys());n[split]+=1;total[split]+=sum(counts.values())
        if doc['id'] in needed:tf[doc['id']]=counts;lengths[doc['id']]=sum(counts.values())
    preparation=time.perf_counter()-start
    dense={}
    for r in map(json.loads,(root/'dense-baseline-runs.jsonl').read_text().splitlines()):dense[(r['query_id'],r['method'])]=r['ranking']
    rows=[]
    for qid,q in queries.items():
        if q['split']!='development':continue
        pool=pools[qid]['candidates'];query_terms=sorted(set(tokens(q['text'])));avg=total['development']/n['development'];method_scores={m:{} for m in ['bm25-max','tfidf-max']}
        for eid in pool:
            scores={m:[] for m in method_scores}
            for span in snapshots[(qid,eid)]['sources']:
                sid=span['source_id'];counts=tf[sid];length=lengths[sid];bm=weighted=0.
                for term in query_terms:
                    frequency=counts.get(term,0)
                    if not frequency:continue
                    idf=math.log(1+(n['development']-df['development'][term]+.5)/(df['development'][term]+.5))
                    bm+=idf*frequency*2.2/(frequency+1.2*(.25+.75*length/max(avg,1e-30)))
                    weighted+=idf*(1+math.log(frequency))/math.sqrt(max(length,1))
                scores['bm25-max'].append(bm);scores['tfidf-max'].append(weighted)
            for method in scores:method_scores[method][eid]=max(scores[method],default=0.)
        rankings={m:sorted(values,key=lambda e:(-values[e],e)) for m,values in method_scores.items()}
        for method in ['qwen-paper-max','qwen-view-max']:rankings[method]=dense[(qid,method)]
        shuffled=sorted(pool);random.Random(qid+':baseline').shuffle(shuffled);rankings['random']=shuffled
        truth=judgments[qid]
        for method,ranking in rankings.items():
            if set(ranking)!=set(pool) or len(ranking)!=len(pool):raise ValueError('Rankings must use exactly the supplied7snapshot candidates')
            positives={e for e,g in truth.items() if g is not None and g>=2};first=next((i+1 for i,e in enumerate(ranking[:5]) if e in positives),None)
            rows.append(dict(query_id=qid,method=method,ranking=ranking,actionable_gain=actionable_gain_interval(ranking,truth,5),graded_gain=snapshot_gain_interval(ranking,truth,5),
                             condensed_ndcg5=condensed_ndcg(ranking,truth,5),known_actionable_query=bool(positives),known_method_rr5=1/first if first else 0.))
    methods=sorted({r['method'] for r in rows});summary={}
    for method in methods:
        selected=[r for r in rows if r['method']==method];positive=[r for r in selected if r['known_actionable_query']];ndcg=[r['condensed_ndcg5'] for r in selected if r['condensed_ndcg5'] is not None]
        summary[method]=dict(queries=len(selected),actionable_gain_lower=float(np.mean([r['actionable_gain']['lower'] for r in selected])),actionable_gain_upper=float(np.mean([r['actionable_gain']['upper'] for r in selected])),
                            mean_judged_returned_fraction=float(np.mean([r['actionable_gain']['judged_fraction_of_returned'] for r in selected])),
                            diagnostic_condensed_ndcg5=float(np.mean(ndcg)) if ndcg else None,ndcg_defined_queries=len(ndcg),known_actionable_queries=len(positive),known_method_mrr5=float(np.mean([r['known_method_rr5'] for r in positive])) if positive else None)
    out=Path('artifacts/fresh-evaluation-baselines-v1');out.mkdir(exist_ok=True)
    report=dict(dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),lexical_statistics_preparation_seconds=preparation,methods=summary,
                warning='Fixed candidate-source-view development silver only. BM25/TF-IDF use split-specific corpus prefix statistics; Qwen cached512token source vectors. Candidate pool selected by dense plus random, not diverse method pooling. Unknown annotations widen bounds, not actual inability. Bounds conditional on silver grades, not semantic confidence; nDCG/MRR diagnostics not independent expert accuracy. No heldout metrics or online latency claim.')
    (out/'evaluation.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='artifacts/dataset-release/fresh-enterprise-eval-v1');run(p.parse_args().dataset)
