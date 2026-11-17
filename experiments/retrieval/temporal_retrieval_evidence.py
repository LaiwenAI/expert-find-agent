"""Compare completed retrieval plus evidence on local temporal proxy."""
from collections import defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.fast_retrieval import FastSearch


def main():
    history=Path('artifacts/corpus-asof-2020-v1');parent=Path('artifacts/local-temporal-author-proxy-v1');previous=Path('artifacts/local-temporal-adaptive-scores-v1')
    meta=json.loads((parent/'summary.json').read_text());payload=(parent/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=meta['results_sha256'] or hashlib.sha256((history/'evidence.jsonl').read_bytes()).hexdigest()!=meta['historical_sha256']:raise ValueError('Changed snapshot')
    cases=[json.loads(s) for s in payload.splitlines()];reference={r['id']:r for r in json.loads((previous/'results.json').read_text())}
    if len(cases)!=200 or len(reference)!=200 or {c['id'] for c in cases}!=set(reference):raise ValueError('Changed query set')
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(history,'artifacts/temporal-candidate-lm-2020-v1','artifacts/temporal-evidence-2020-v1.sqlite')
    fast=FastSearch(history,'artifacts/temporal-evidence-2020-v1.sqlite');fast.prepare_adaptive_scoring();init=time.perf_counter()-start
    profiles=defaultdict(list)
    for _,(eid,expert) in fast.metadata.items():profiles[expert].append(eid)
    available=sorted(pipeline.ranker.model.available);rows=[]
    for i,case in enumerate(cases):
        arms={}
        for method in (('adaptive50','global') if i%2==0 else ('global','adaptive50')):
            start=time.perf_counter()
            if method=='adaptive50':
                pool=[r['expert']['id'] for r in pipeline.ranker.search(case['title'],50)['results']]
                source_ids=[eid for expert in pool for eid in profiles[expert]]
                scores=fast.adaptive_document_scores(case['title'],source_ids)['scores']
            else:
                pool=available;scores=fast.document_scores(case['title'])
            values={expert:max((scores.get(eid,0.) for eid in profiles[expert]),default=0.) for expert in pool}
            ranking=sorted(pool,key=lambda e:(-values[e],e))[:10]
            if ranking!=reference[case['id']]['methods'][method]['ranking']:raise ValueError('Prior ranking changed')
            evidence=fast.evidence_for_selective(case['title'],ranking,1)
            for expert in ranking:
                if not evidence[expert] or evidence[expert][0]['expert_id']!=expert or abs(evidence[expert][0]['score']-values[expert])>1e-10:
                    raise ValueError('Incorrect source owner or score')
            elapsed=time.perf_counter()-start;truth=set(case['known_author_ids'])
            arms[method]=dict(ranking=ranking,seconds=elapsed,hit=int(bool(set(ranking)&truth)),evidence_count=sum(len(evidence[e]) for e in ranking))
        rows.append(dict(id=case['id'],arms=arms))
    out=Path('artifacts/local-temporal-evidence-comparison-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(queries=len(rows),initialization_seconds=init,all_ranking_and_evidence_checks=True,
                mean_seconds={m:statistics.mean(r['arms'][m]['seconds'] for r in rows) for m in ('adaptive50','global')},
                hit_at_10={m:statistics.mean(r['arms'][m]['hit'] for r in rows) for m in ('adaptive50','global')},
                returned_evidence={m:sum(r['arms'][m]['evidence_count'] for r in rows) for m in ('adaptive50','global')},
                parent_results_sha256=meta['results_sha256'],scope='Post-hoc one-pass retrieval plus one evidence item per result on a previously studied local author-recovery proxy. Same compiled score and evidence policy; model reasoning, index construction and initialization excluded. Evidence ownership and score equality do not establish semantic support, identity, personal contribution, or industrial applicability. Timing is not a latency distribution.')
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));fast.db.close();pipeline.evidence.db.close()


if __name__=='__main__':main()
