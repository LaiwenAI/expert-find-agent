"""Compare adaptive and selective source-scoring paths on CoF SIGIR."""
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch


def main():
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-sigir-selective-rerank-v1')
    old=json.loads((prior/'summary.json').read_text());old_rows={r['query_id']:r for r in json.loads((prior/'results.json').read_text())}
    if any(hashlib.sha256((root/n).read_bytes()).hexdigest()!=v for n,v in old['input_sha256'].items()):raise ValueError('Changed data')
    ids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    docs={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in ids}
    profiles={eid:sorted(set(r['papers'])) for eid,r in reviewers.items()};model=CompiledCandidateLM(ExpertLanguageModels(docs,profiles));available=set(model.available)
    if available!=set(reviewers) or any(set(labels[t][qid])!=available for t in labels for qid in labels[t]):raise ValueError('Judgments not complete')
    search=FastSearch(root/'unique-profile-corpus',prior/'selective.sqlite');start=time.perf_counter();search.prepare_adaptive_scoring();prepare_seconds=time.perf_counter()-start
    methods=('adaptive20','score20','adaptive50','score50','global');rows=[]
    for i,qid in enumerate(sorted(labels['soft'])):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip();results={}
        for method in methods[i%5:]+methods[:i%5]:
            start=time.perf_counter();route=None
            if method=='global':
                values=search.document_scores(query);pool=sorted(available)
            else:
                n=20 if method.endswith('20') else 50;pool=[r['id'] for r in certified_topk(model,query,n)['results']]
                sources=sorted({pid for eid in pool for pid in profiles[eid]})
                if method.startswith('adaptive'):
                    selected=search.adaptive_document_scores(query,sources);values=selected['scores'];route=selected['route']
                else:values=search.selective_document_scores(query,sources)
            scores={eid:max((values.get(pid,0.) for pid in profiles[eid]),default=0.) for eid in pool}
            ranking=sorted(pool,key=lambda eid:(-scores[eid],eid))[:10];elapsed=time.perf_counter()-start
            if method!='global' and ranking!=old_rows[qid]['rankings']['candidate'+method[-2:]+'_bm25']:raise ValueError('Changed shortlist ranking')
            if method=='global' and ranking!=old_rows[qid]['rankings']['global_bm25']:raise ValueError('Changed global ranking')
            results[method]=dict(seconds=elapsed,route=route,ranking=ranking,metrics={t:precision_metrics(ranking,labels[t][qid]) for t in labels})
        rows.append(dict(query_id=qid,methods=results))
    out=Path('artifacts/cof-sigir-adaptive-scores-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(queries=len(rows),all_rankings_match_previous=True,posting_count_preparation_seconds=prepare_seconds,
                routes={m:{route:sum(r['methods'][m]['route']==route for r in rows) for route in ('global','selective')} for m in ('adaptive20','adaptive50')},
                mean_seconds={m:statistics.mean(r['methods'][m]['seconds'] for r in rows) for m in methods},
                metrics={t:{m:{k:statistics.mean(r['methods'][m]['metrics'][t][k] for r in rows) for k in ('p@5','ndcg@10')} for m in methods} for t in labels},
                prior_summary_sha256=hashlib.sha256((prior/'summary.json').read_bytes()).hexdigest(),
                scope='Post-hoc repeated single rotating-order timings on same CoF SIGIR data. Adaptive and score-only paths omit evidence-body materialization and explanation; not end-to-end search. Index construction and initialization excluded. Fully judged reviewer relevance, not industrial fit. Posting-count heuristic is fixed without per-query tuning and does not guarantee optimal routing. No quality improvement or latency distribution established.')
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));search.db.close()


if __name__=='__main__':main()
