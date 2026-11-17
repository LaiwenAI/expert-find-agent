"""Compare exact resident posting-intersection directions on private development needs."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query source')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'candidate_pool':50,'return_k':20,'evidence_per_expert':1,'arm_order':'alternating',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Reused enterprise development queries; exact same candidate ranking, source scores and evidence required. No protected split, reasoning or independent relevance labels.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');init=time.perf_counter()-start;rows=[]
    for i,q in enumerate(queries):
        results={};seconds={}
        for arm in (('fixed','adaptive') if i%2==0 else ('adaptive','fixed')):
            engine.resident.adaptive_intersection=(arm=='adaptive');start=time.perf_counter()
            results[arm]=engine.search_reranked(q['text'],k=20,candidates=50,evidence_per_expert=1)
            seconds[arm]=time.perf_counter()-start
        if results['fixed']['results']!=results['adaptive']['results'] or results['fixed']['source_snapshot_sha256']!=results['adaptive']['source_snapshot_sha256']:
            raise ValueError('Changed scores, ranking or evidence')
        rows.append({'query_id':q['id'],'seconds':seconds})
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'identical_complete_results':len(rows),'initialization_seconds':init,
            'mean_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in ('fixed','adaptive')},
            'paired_seconds':paired_interval([r['seconds']['adaptive']-r['seconds']['fixed'] for r in rows]),
            'adaptive_faster_queries':sum(r['seconds']['adaptive']<r['seconds']['fixed'] for r in rows),
            'corpus_snapshot_sha256':engine.ranker.snapshot,'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),
            'scope':protocol['scope']+' One alternating timing per query; unadjusted bootstrap assumes independent queries. Initialization, model reasoning, production concurrency and index construction excluded from query times; query text and IDs remain private.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
