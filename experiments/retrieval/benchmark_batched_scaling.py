"""Profile batched source scoring across candidate budgets and query lengths."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import tokens
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_shape(resident, terms, source_ids, limit=100000):
    positions=len(set(source_ids));pending=0;largest=0;groups=0;long_lists=0
    for term in sorted(set(terms)):
        offsets=resident.offsets.get(term)
        if offsets is None or not positions:continue
        count=offsets[1]-offsets[0]
        if count<positions and count<=limit:
            if pending+count>limit:
                largest=max(largest,pending);groups+=bool(pending);pending=0
            pending+=count
        else:
            largest=max(largest,pending);groups+=bool(pending);pending=0;long_lists+=1
    largest=max(largest,pending);groups+=bool(pending)
    return dict(selected_sources=positions,max_batched_postings=largest,batches=groups,long_lists=long_lists)


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query file')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({r['id'] for r in queries})!=80:raise ValueError('Changed development cohort')
    files=[Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/retrieval.py')]
    protocol=dict(queries=len(queries),query_sha256=sha(data/'queries.jsonl'),code_sha256={str(p):sha(p) for p in files},candidate_budgets=[10,20,50],return_k='min(candidate budget,20)',warmups_per_arm=1,repeats_per_arm=2,batch_limit=100000,order='alternate query and repeat',
        scope='Reused development needs; paired complete result equivalence. Batch counts are analytically computed posting counts, not memory measurements. No independent relevance labels.')
    output.mkdir(parents=True,exist_ok=False);(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident-adaptive',share_query_tokens=True);init=time.perf_counter()-start;rows=[]
    for i,q in enumerate(queries):
        term_list=tokens(q['text'])
        for budget in (10,20,50):
            candidate=engine.ranker.search(q['text'],budget,_tokens=term_list)
            ids=[row['expert']['id'] for row in candidate['results']]
            source_ids=[eid for expert in ids for eid in engine.evidence_by_expert[expert]]
            shape=batch_shape(engine.resident,term_list,source_ids)
            seconds={'adaptive':[],'batched':[]};reference=None
            for repeat in range(-1,2):
                for arm in (('adaptive','batched') if (i+repeat+budget//10)%2==0 else ('batched','adaptive')):
                    engine.batch_resident_postings=arm=='batched';start=time.perf_counter()
                    result=engine.search_reranked(q['text'],k=min(20,budget),candidates=budget,evidence_per_expert=1)
                    elapsed=time.perf_counter()-start;observed=(result['results'],result['source_snapshot_sha256'])
                    if reference is None:reference=observed
                    elif observed!=reference:raise ValueError('Changed complete result')
                    if repeat>=0:seconds[arm].append(elapsed)
            rows.append(dict(query_id=q['id'],budget=budget,unique_terms=len(set(term_list)),shape=shape,seconds=seconds))
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    budget_stats={}
    for budget in (10,20,50):
        group=[r for r in rows if r['budget']==budget];means={a:[statistics.mean(r['seconds'][a]) for r in group] for a in ('adaptive','batched')}
        differences=[a-b for a,b in zip(means['batched'],means['adaptive'])]
        ordered=sorted(group,key=lambda r:(r['unique_terms'],r['query_id']))
        strata=[]
        for quartile in range(4):
            subset=ordered[quartile*20:(quartile+1)*20]
            strata.append(dict(queries=len(subset),unique_terms_range=[subset[0]['unique_terms'],subset[-1]['unique_terms']],
                mean_difference_seconds=statistics.mean(statistics.mean(x['seconds']['batched'])-statistics.mean(x['seconds']['adaptive']) for x in subset)))
        budget_stats[str(budget)]=dict(mean_seconds={a:statistics.mean(means[a]) for a in means},batched_faster_query_means=sum(d<0 for d in differences),paired_seconds=paired_interval(differences),
            max_observed_slowdown_seconds=max(differences),mean_selected_sources=statistics.mean(r['shape']['selected_sources'] for r in group),max_batched_postings=max(r['shape']['max_batched_postings'] for r in group),
            mean_batches=statistics.mean(r['shape']['batches'] for r in group),query_length_quartiles=strata)
    summary=dict(queries=80,configs=len(rows),matched_complete_results=len(rows)*6,initialization_seconds=init,budgets=budget_stats,
        protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'private-results.json'),corpus_snapshot_sha256=engine.ranker.snapshot,
        scope=protocol['scope']+' Two dependent timings per query; query-resampled intervals are unadjusted for comparisons. Quartiles and maxima are descriptive; one machine, no concurrent load, initialization or model reasoning.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
