"""Compare batched and adaptive exact candidate source scoring."""
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
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed development cohort')
    output.mkdir(parents=True,exist_ok=False)
    files=(Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/compiled_expert_search.py'),Path('src/expert_search/retrieval.py'))
    protocol=dict(queries=len(queries),query_file_sha256=sha(data/'queries.jsonl'),code_sha256={str(p):sha(p) for p in files},pool=50,k=20,evidence_per_expert=1,warmups=1,repeats=3,order='alternating by query and repeat',batch_limit=100000,
        scope='Reused enterprise development needs. Identical complete candidate ranking, scores and evidence required. No independent relevance or expert ability labels.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident-adaptive',share_query_tokens=True);init=time.perf_counter()-start
    rows=[]
    for i,q in enumerate(queries):
        times={'adaptive':[],'batched':[]};reference=None
        for repeat in range(-1,3):
            for arm in (('adaptive','batched') if (i+repeat)%2==0 else ('batched','adaptive')):
                engine.batch_resident_postings=arm=='batched'
                start=time.perf_counter();result=engine.search_reranked(q['text'],k=20,candidates=50,evidence_per_expert=1);elapsed=time.perf_counter()-start
                observed=(result['results'],result['source_snapshot_sha256'])
                if reference is None:reference=observed
                elif observed!=reference:raise ValueError('Changed complete result')
                if repeat>=0:times[arm].append(elapsed)
        rows.append(dict(query_id=q['id'],seconds=times))
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    means={a:[statistics.mean(r['seconds'][a]) for r in rows] for a in ('adaptive','batched')}
    summary=dict(queries=len(rows),matched_complete_results=len(rows)*8,initialization_seconds=init,
        mean_seconds={a:statistics.mean(means[a]) for a in means},empirical_p95_seconds={a:float(np.quantile([x for r in rows for x in r['seconds'][a]],.95)) for a in means},
        batched_faster_query_means=sum(a<b for a,b in zip(means['batched'],means['adaptive'])),paired_seconds=paired_interval([a-b for a,b in zip(means['batched'],means['adaptive'])]),
        protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'private-results.json'),corpus_snapshot_sha256=engine.ranker.snapshot,
        scope=protocol['scope']+' Three correlated timings per query; bootstrap resamples query means and assumes query independence. Excludes initialization, index build, model reasoning and concurrent load; empirical p95 is not a production percentile.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
