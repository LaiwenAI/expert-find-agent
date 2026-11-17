"""Compare exact posting-intersection directions on public CoF reviewer queries."""
import json
from pathlib import Path
import statistics
import time
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import sha,verify_prior


def main():
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2');hashes=verify_prior(root,papers,reviewers,prior)
    ids=sorted(labels['soft']);out=Path('artifacts/cof-resident-intersections-v1');out.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':ids,'input_sha256':hashes,'candidate_pool':50,'return_k':20,'evidence_per_expert':1,'arm_order':'alternating',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Reused CoF reviewer-matching queries; same fixed corpus, rankings, scores and evidence required. No industrial fit or independent evidence labels.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident');init=time.perf_counter()-start;rows=[]
    for i,qid in enumerate(ids):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip();results={};times={}
        for arm in (('fixed','adaptive') if i%2==0 else ('adaptive','fixed')):
            engine.resident.adaptive_intersection=(arm=='adaptive');start=time.perf_counter()
            results[arm]=engine.search_reranked(query,k=20,candidates=50,evidence_per_expert=1);times[arm]=time.perf_counter()-start
        if results['fixed']['results']!=results['adaptive']['results'] or results['fixed']['source_snapshot_sha256']!=results['adaptive']['source_snapshot_sha256']:raise ValueError('Changed complete results')
        ranking=[r['expert']['id'] for r in results['fixed']['results']]
        rows.append({'query_id':qid,'seconds':times,'metrics':{t:precision_metrics(ranking,labels[t][qid]) for t in labels}})
    engine.evidence.db.close();(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'identical_complete_results':len(rows),'initialization_seconds':init,'mean_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in ('fixed','adaptive')},
            'adaptive_faster_queries':sum(r['seconds']['adaptive']<r['seconds']['fixed'] for r in rows),
            'paired_seconds':paired_interval([r['seconds']['adaptive']-r['seconds']['fixed'] for r in rows]),
            'protocol_sha256':sha(out/'protocol.json'),'results_sha256':sha(out/'results.json'),
            'scope':protocol['scope']+' One alternating measurement per query; bootstrap assumes independent queries and is unadjusted. Query time excludes initialization and reasoning; owner copies change document frequencies.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
