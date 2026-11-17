"""Repeat alternating exact intersection timings with per-query paired analysis."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query source')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'candidate_pool':50,'return_k':20,'evidence_per_expert':1,
              'warmups_per_arm_per_query':1,'timed_repeats_per_arm_per_query':5,'arm_order':'alternating by query and repeat',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Reused development needs; fixed versus adaptive exact intersection. Protected queries and model reasoning excluded. All complete results must match.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');init=time.perf_counter()-start;rows=[]
    for i,q in enumerate(queries):
        times={'fixed':[],'adaptive':[]};reference=None
        for repeat in range(-1,5):
            order=('fixed','adaptive') if (i+repeat)%2==0 else ('adaptive','fixed')
            for arm in order:
                engine.resident.adaptive_intersection=(arm=='adaptive');start=time.perf_counter()
                result=engine.search_reranked(q['text'],k=20,candidates=50,evidence_per_expert=1);elapsed=time.perf_counter()-start
                observed=(result['results'],result['source_snapshot_sha256'])
                if reference is None:reference=observed
                elif reference!=observed:raise ValueError('Changed complete result')
                if repeat>=0:times[arm].append(elapsed)
        rows.append({'query_id':q['id'],'seconds':times})
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    per_query={a:[statistics.mean(r['seconds'][a]) for r in rows] for a in ('fixed','adaptive')}
    distribution={a:{'mean_seconds':statistics.mean(per_query[a]),'median_observation_seconds':float(np.median([v for r in rows for v in r['seconds'][a]])),
                      'empirical_p95_observation_seconds':float(np.quantile([v for r in rows for v in r['seconds'][a]],.95))} for a in ('fixed','adaptive')}
    report={'queries':len(rows),'timed_observations_per_arm':len(rows)*5,'matching_complete_results':len(rows)*12,
            'initialization_seconds':init,'distribution':distribution,'adaptive_faster_query_means':sum(a<b for a,b in zip(per_query['adaptive'],per_query['fixed'])),
            'paired_query_mean_seconds':paired_interval([a-b for a,b in zip(per_query['adaptive'],per_query['fixed'])]),
            'corpus_snapshot_sha256':engine.ranker.snapshot,'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),
            'scope':protocol['scope']+' Five correlated repeat observations per query; the bootstrap resamples query means, assuming query independence. Empirical percentiles pool repeated measurements and are not production latency percentiles. Timings exclude initialization, model inference, concurrent load, and index construction. Query IDs remain private.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
