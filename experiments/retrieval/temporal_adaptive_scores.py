"""Compare adaptive shortlist scoring on the frozen local temporal proxy."""
from collections import Counter,defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.fast_retrieval import FastSearch


def main():
    history=Path('artifacts/corpus-asof-2020-v1');parent=Path('artifacts/local-temporal-author-proxy-v1')
    lm=Path('artifacts/temporal-candidate-lm-2020-v1');db=Path('artifacts/temporal-evidence-2020-v1.sqlite')
    summary=json.loads((parent/'summary.json').read_text());payload=(parent/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=summary['results_sha256'] or hashlib.sha256((history/'evidence.jsonl').read_bytes()).hexdigest()!=summary['historical_sha256']:raise ValueError('Frozen data changed')
    cases=[json.loads(x) for x in payload.splitlines()]
    if len(cases)!=200 or len({x['id'] for x in cases})!=200:raise ValueError('Unexpected query set')
    pipeline=CandidateEvidencePipeline(history,lm,db);fast=FastSearch(history,db);start=time.perf_counter();fast.prepare_adaptive_scoring();setup=time.perf_counter()-start
    source_by_expert=defaultdict(list)
    for _,(eid,expert) in fast.metadata.items():source_by_expert[expert].append(eid)
    available=sorted(pipeline.ranker.model.available)
    if set(available)-set(source_by_expert):raise ValueError('Indexed candidate lacks evidence')
    methods=('selective20','adaptive20','selective50','adaptive50','global');results=[]
    for i,case in enumerate(cases):
        methods_result={};truth=set(case['known_author_ids'])
        for method in methods[i%5:]+methods[:i%5]:
            start=time.perf_counter();route=None
            if method=='global':
                scores=fast.document_scores(case['title']);pool=available
            else:
                size=20 if method.endswith('20') else 50
                pool=[r['expert']['id'] for r in pipeline.ranker.search(case['title'],size)['results']]
                ids=[eid for expert in pool for eid in source_by_expert[expert]]
                if method.startswith('adaptive'):
                    routed=fast.adaptive_document_scores(case['title'],ids);scores=routed['scores'];route=routed['route']
                else:scores=fast.selective_document_scores(case['title'],ids)
            values={expert:max((scores.get(eid,0.) for eid in source_by_expert[expert]),default=0.) for expert in pool}
            ranking=sorted(pool,key=lambda expert:(-values[expert],expert))[:10]
            elapsed=time.perf_counter()-start;positions=[i for i,e in enumerate(ranking,1) if e in truth]
            methods_result[method]=dict(ranking=ranking,seconds=elapsed,route=route,hit=int(bool(positions)),mrr=1/min(positions) if positions else 0.)
        for n in (20,50):
            if methods_result['adaptive'+str(n)]['ranking']!=methods_result['selective'+str(n)]['ranking']:raise ValueError('Scoring route changed ranking')
        if methods_result['global']['ranking']!=case['methods']['bm25-max']['ranking']:raise ValueError('Global ranking changed')
        results.append(dict(id=case['id'],methods=methods_result))
    out=Path('artifacts/local-temporal-adaptive-scores-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(queries=len(results),posting_count_preparation_seconds=setup,rankings_equal=True,
                routes={m:dict(Counter(r['methods'][m]['route'] for r in results)) for m in ('adaptive20','adaptive50')},
                mean_seconds={m:statistics.mean(r['methods'][m]['seconds'] for r in results) for m in methods},
                hit_at_10={m:statistics.mean(r['methods'][m]['hit'] for r in results) for m in methods},
                mrr_at_10={m:statistics.mean(r['methods'][m]['mrr'] for r in results) for m in methods},
                parent_results_sha256=summary['results_sha256'],history_sha256=summary['historical_sha256'],
                scope='Post-hoc 200-query local temporal author-recovery proxy. Teacher-paper associations and identities unverified; other authors unjudged, absent historical authors excluded. Single rotating-order timing per query excludes initialization, index construction, evidence bodies and model reasoning. No quality, latency distribution, or industrial-fit guarantee.')
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    fast.db.close();pipeline.evidence.db.close()


if __name__=='__main__':main()
