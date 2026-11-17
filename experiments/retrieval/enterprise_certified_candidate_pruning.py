"""Evaluate exact owner-bound pruning on private enterprise development needs."""
import argparse
import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.expert_upper_bound import ExpertTermUpperBounds,certified_candidate_max
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query source')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed development cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/expert_upper_bound.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'queries':len(queries),'queries_sha256':sha(data/'queries.jsonl'),'pool':50,'seed':20,'k':5,'evidence_per_expert':1,'arm_order':'alternating',
              'code_sha256':{str(p):sha(p) for p in code},'scope':'Reused private development needs; exactness relative to the same 50-candidate max-BM25 path, not expert relevance or global top-k. No protected queries.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');pipeline_init=time.perf_counter()-start
    print('Resident index initialized',flush=True);rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bounds=ExpertTermUpperBounds(engine.resident,engine.evidence_by_expert);rss_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Bounds initialized',flush=True);rows=[]
    for i,q in enumerate(queries):
        attempts={}
        for arm in (('baseline','certified') if i%2==0 else ('certified','baseline')):
            start=time.perf_counter()
            result=(engine.search_reranked(q['text'],k=5,candidates=50,evidence_per_expert=1) if arm=='baseline' else certified_candidate_max(engine,bounds,q['text']))
            attempts[arm]=(result,time.perf_counter()-start)
        baseline,bt=attempts['baseline'];cert,ct=attempts['certified']
        if baseline['results']!=cert['results'] or baseline['source_snapshot_sha256']!=cert['source_snapshot_sha256']:raise ValueError('Certified result differs')
        rows.append({'query_id':q['id'],'baseline_seconds':bt,'certified_seconds':ct,'pool_size':cert['pool_size'],'scored_candidates':cert['scored_candidates'],'pruned_candidates':cert['pruned_candidates']})
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'identical_complete_results':len(rows),'mean_scored_candidates':statistics.mean(r['scored_candidates'] for r in rows),
            'queries_with_pruning':sum(r['pruned_candidates']>0 for r in rows),'mean_pruned_candidates':statistics.mean(r['pruned_candidates'] for r in rows),
            'mean_baseline_seconds':statistics.mean(r['baseline_seconds'] for r in rows),'mean_certified_seconds':statistics.mean(r['certified_seconds'] for r in rows),
            'paired_seconds':paired_interval([r['certified_seconds']-r['baseline_seconds'] for r in rows]),'pipeline_init_seconds':pipeline_init,
            'bound_init_seconds':bounds.initialization_seconds,'bound_numeric_array_bytes':bounds.array_bytes,
            'process_peak_rss_before_bound':rss_before,'process_peak_rss_after_bound':rss_after,'peak_rss_units':'bytes on macOS; KiB on Linux',
            'corpus_snapshot_sha256':engine.ranker.snapshot,'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),
            'scope':protocol['scope']+' One alternating measurement per query; bootstrap assumes independent queries and is unadjusted. Query timings omit initialization and model reasoning. Memory numbers are process peaks, not an isolated increment; array bytes exclude Python overhead. Source data and query IDs remain private.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
