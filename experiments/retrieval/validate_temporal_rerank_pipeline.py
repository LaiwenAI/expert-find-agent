"""Replay the reusable rerank pipeline against frozen temporal comparisons."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def main():
    parent=Path('artifacts/local-temporal-author-proxy-v1');comparison=Path('artifacts/local-temporal-evidence-comparison-v1')
    summary=json.loads((parent/'summary.json').read_text());payload=(parent/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=summary['results_sha256']:raise ValueError('Changed query set')
    queries=[json.loads(s) for s in payload.splitlines()];reference={r['id']:r for r in json.loads((comparison/'results.json').read_text())}
    if len(queries)!=200 or {q['id'] for q in queries}!=set(reference):raise ValueError('Changed cohort')
    pipeline=CandidateEvidencePipeline('artifacts/corpus-asof-2020-v1','artifacts/temporal-candidate-lm-2020-v1','artifacts/temporal-evidence-2020-v1.sqlite')
    times=[];routes=Counter();hits=0;source_count=0
    for q in queries:
        result=pipeline.search_reranked(q['title'],k=10,candidates=50,evidence_per_expert=1)
        ranking=[row['expert']['id'] for row in result['results']]
        if ranking!=reference[q['id']]['arms']['adaptive50']['ranking']:raise ValueError('Pipeline ranking differs')
        for row in result['results']:
            if len(row['evidence'])!=1 or row['evidence'][0]['expert_id']!=row['expert']['id'] or abs(row['evidence'][0]['score']-row['score'])>1e-10:
                raise ValueError('Evidence owner or score differs')
            source_count+=1
        hits+=bool(set(ranking)&set(q['known_author_ids']));times.append(result['timing']['total_seconds']);routes[result['route']]+=1
    pipeline.evidence.db.close()
    report=dict(queries=len(queries),matching_rankings=len(queries),evidence_checked=source_count,known_author_hits=hits,
                route_counts=dict(routes),mean_seconds=statistics.mean(times),parent_sha256=summary['results_sha256'],
                scope='Post-hoc integration replay on local previously studied author-recovery proxy. One timing per query includes lazy route setup in first query but excludes index construction and pipeline initialization. Source association and lexical score do not validate actual identity, individual role, semantic support, or enterprise fit.')
    out=Path('artifacts/local-temporal-rerank-pipeline-v1');out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
