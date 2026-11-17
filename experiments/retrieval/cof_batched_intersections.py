"""Compare exact candidate scoring on full and title-only public queries."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load
from expert_search.retrieval import tokens
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import verify_prior
from experiments.retrieval.benchmark_batched_scaling import batch_shape


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2');hashes=verify_prior(root,papers,reviewers,prior)
    ids=sorted(labels['soft'])
    if len(ids)!=73 or set(ids)!=set(labels['hard']):raise ValueError('Changed query cohort')
    files=[Path(__file__),Path('experiments/retrieval/benchmark_batched_scaling.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/retrieval.py')]
    protocol=dict(query_ids=ids,input_sha256=hashes,code_sha256={str(p):sha(p) for p in files},candidate_pool=50,return_k=20,evidence_per_expert=1,arms=['adaptive','batched'],views=['full','title'],warmups=1,repeats=3,
        scope='Previously examined CoF reviewer queries, with owner-copy corpus. Title-only truncates the query and does not retain the full task. Check within-view complete result equality only; no industrial fit, source entailment or cross-view quality claim.')
    output.mkdir(parents=True,exist_ok=False);(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident-adaptive',share_query_tokens=True);init=time.perf_counter()-start;rows=[]
    for i,qid in enumerate(ids):
        paper=papers[qid];title=paper.get('title','').strip();full=(title+'\n'+paper.get('abstract','')).strip()
        if not title or not full:raise ValueError('Missing query title')
        for view,query in (('full',full),('title',title)):
            term_list=tokens(query);candidate=engine.ranker.search(query,50,_tokens=term_list)
            ids_for_pool=[r['expert']['id'] for r in candidate['results']]
            source_ids=[eid for expert in ids_for_pool for eid in engine.evidence_by_expert[expert]]
            shape=batch_shape(engine.resident,term_list,source_ids)
            seconds={'adaptive':[],'batched':[]};reference=None
            for repeat in range(-1,3):
                for arm in (('adaptive','batched') if (i+repeat+(view=='title'))%2==0 else ('batched','adaptive')):
                    engine.batch_resident_postings=arm=='batched';start=time.perf_counter()
                    result=engine.search_reranked(query,k=20,candidates=50,evidence_per_expert=1);elapsed=time.perf_counter()-start
                    actual=(result['results'],result['source_snapshot_sha256'])
                    if reference is None:reference=actual
                    elif actual!=reference:raise ValueError('Changed ranking, scores, or source views')
                    if repeat>=0:seconds[arm].append(elapsed)
            rows.append(dict(query_id=qid,view=view,unique_terms=len(set(term_list)),shape=shape,seconds=seconds))
    engine.evidence.db.close();(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    reports={}
    for view in ('full','title'):
        group=[r for r in rows if r['view']==view];means={a:[statistics.mean(r['seconds'][a]) for r in group] for a in ('adaptive','batched')}
        delta=[a-b for a,b in zip(means['batched'],means['adaptive'])]
        reports[view]=dict(queries=len(group),mean_unique_terms=statistics.mean(r['unique_terms'] for r in group),mean_seconds={a:statistics.mean(means[a]) for a in means},
            batched_faster_query_means=sum(d<0 for d in delta),paired_seconds=paired_interval(delta),max_observed_slowdown_seconds=max(delta),max_batched_postings=max(r['shape']['max_batched_postings'] for r in group))
    result=dict(queries=len(ids),views=len(rows),matched_complete_results=len(rows)*8,initialization_seconds=init,reports=reports,protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'results.json'),
        scope=protocol['scope']+' Query-bootstrap intervals are unadjusted and assume query independence. Three dependent repeats, one host; no index initialization, reasoning, load or serving in query time. Posting count is not peak memory.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
