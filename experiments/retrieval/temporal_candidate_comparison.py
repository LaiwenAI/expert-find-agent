"""Compare candidate-first and compiled lexical retrieval on a local proxy."""
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.fast_retrieval import FastSearch

HISTORY=Path('artifacts/corpus-asof-2020-v1')
PARENT=Path('artifacts/local-temporal-author-proxy-v1')
LM=Path('artifacts/temporal-candidate-lm-2020-v1')
DB=Path('artifacts/temporal-evidence-2020-v1.sqlite')
OUT=Path('artifacts/local-temporal-candidate-comparison-v1')


def main():
    manifest=json.loads((PARENT/'summary.json').read_text());payload=(PARENT/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['results_sha256'] or hashlib.sha256((HISTORY/'evidence.jsonl').read_bytes()).hexdigest()!=manifest['historical_sha256']:
        raise ValueError('Frozen proxy changed')
    rows=[json.loads(line) for line in payload.splitlines()]
    if len(rows)!=200 or len({r['id'] for r in rows})!=200:raise ValueError('Invalid query set')
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(HISTORY,LM,DB);candidate_init=time.perf_counter()-start
    start=time.perf_counter();lexical=FastSearch(HISTORY,DB);lexical_init=time.perf_counter()-start
    methods=('compiled-bm25-max','candidate-lm','candidate50-bm25-vote');results=[]
    for i,row in enumerate(rows):
        truth=set(row['known_author_ids']);value={}
        for method in methods[i%3:]+methods[:i%3]:
            started=time.perf_counter()
            if method=='compiled-bm25-max':
                ids=[r['id'] for r in lexical.search(row['title'],10,'bm25-max')]
            elif method=='candidate-lm':
                ids=[r['expert']['id'] for r in pipeline.search(row['title'],10,3)['results']]
            else:
                pool=pipeline.search(row['title'],50,5)['results']
                ranked=sorted(pool,key=lambda r:(-sum(e['score'] for e in r['evidence']),r['expert']['id']))
                ids=[r['expert']['id'] for r in ranked if any(e['score']>0 for e in r['evidence'])][:10]
            elapsed=time.perf_counter()-started;positions=[rank for rank,eid in enumerate(ids,1) if eid in truth]
            value[method]=dict(ranking=ids,seconds=elapsed,hit_at_10=int(bool(positions)),mrr_at_10=1/min(positions) if positions else 0,
                               known_author_recall_at_10=len(set(ids)&truth)/len(truth))
        results.append(dict(id=row['id'],methods=value))
    summary=dict(queries=len(results),candidate_initialization_seconds=candidate_init,lexical_initialization_seconds=lexical_init,
                 exact_bm25_order_agreement=sum(r['methods']['compiled-bm25-max']['ranking']==source['methods']['bm25-max']['ranking'] for r,source in zip(results,rows)),
                 methods={m:{k:statistics.mean(r['methods'][m][k] for r in results) for k in ('seconds','hit_at_10','mrr_at_10','known_author_recall_at_10')} for m in methods},
                 parent_results_sha256=manifest['results_sha256'],history_sha256=manifest['historical_sha256'],
                 scope='Post-hoc comparison on previously examined local author-recovery proxy. Known authors reflect unverified source associations; all other authors unjudged, unavailable authors excluded, semantic leakage may persist. One rotating-order timing per query/method excludes index construction, initialization and reasoning; latency is local, not a distribution or end-to-end measurement.')
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    pipeline.evidence.db.close();lexical.db.close()


if __name__=='__main__':main()
