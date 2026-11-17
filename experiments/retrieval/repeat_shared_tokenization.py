"""Paired replay of shared query tokenization on development needs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
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
    files=(Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/compiled_expert_search.py'),Path('src/expert_search/certified_topk.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/retrieval.py'))
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'code_sha256':{str(p):sha(p) for p in files},'candidate_pool':50,'return_k':20,'evidence_per_expert':1,'warmups_per_arm':1,'repeats_per_arm':5,'order':'alternating by query and repeat',
              'scope':'Reused development needs; exact adaptive resident ranking and evidence, with versus without one shared query tokenization. No relevance or capability judgment.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident-adaptive');initialization=time.perf_counter()-start;rows=[]
    for i,q in enumerate(queries):
        times={'separate':[],'shared':[]};reference=None
        for repeat in range(-1,5):
            for arm in (('separate','shared') if (i+repeat)%2==0 else ('shared','separate')):
                engine.share_query_tokens=arm=='shared';start=time.perf_counter()
                result=engine.search_reranked(q['text'],k=20,candidates=50,evidence_per_expert=1);elapsed=time.perf_counter()-start
                observed=(result['results'],result['source_snapshot_sha256'])
                if reference is None:reference=observed
                elif observed!=reference:raise ValueError('Changed complete result')
                if repeat>=0:times[arm].append(elapsed)
        rows.append({'query_id':q['id'],'seconds':times})
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    means={a:[statistics.mean(r['seconds'][a]) for r in rows] for a in ('separate','shared')}
    report={'queries':len(rows),'matched_complete_results':len(rows)*12,'initialization_seconds':initialization,
            'distribution':{a:{'mean_seconds':statistics.mean(means[a]),'empirical_p95_observation_seconds':float(np.quantile([t for r in rows for t in r['seconds'][a]],.95))} for a in means},
            'shared_faster_query_means':sum(a<b for a,b in zip(means['shared'],means['separate'])),
            'paired_query_mean_seconds':paired_interval([a-b for a,b in zip(means['shared'],means['separate'])]),
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),'corpus_snapshot_sha256':engine.ranker.snapshot,
            'scope':protocol['scope']+' Five correlated timings per query; bootstrap resamples query means assuming query independence. Pooled P95 is not production latency. Initialization, index build, model inference and concurrency are excluded.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
