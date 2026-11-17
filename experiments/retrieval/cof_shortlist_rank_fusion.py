"""Compare within-shortlist fusion against source-max on reused CoF SIGIR queries."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.channel_fusion import fuse
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import sha, verify_prior


def main(output):
    root,papers,reviewers,labels=load('SIGIR')
    prior=Path('artifacts/cof-owned-vote-quality-v2')
    hashes=verify_prior(root,papers,reviewers,prior)
    ids=sorted(labels['soft'])
    if any(set(labels[t][q])!=set(reviewers) for t in labels for q in ids):
        raise ValueError('Candidate judgments must be complete')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':ids,'input_sha256':hashes,'candidate_pool':50,'return_k':20,
              'fusion':'RRF constant 60 within candidate pool',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/channel_fusion.py'))},
              'scope':'Post-hoc reused public queries; owner-linked source copies alter document frequencies. Candidate restriction excludes all other reviewers. No independent enterprise relevance or evidence entailment.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident-adaptive');initialization=time.perf_counter()-start
    rows=[]
    for qid in ids:
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        start=time.perf_counter();engine._check_resident_snapshot();candidate=engine.ranker.search(query,50)
        pool=[r['expert']['id'] for r in candidate['results']]
        sources=[s for eid in pool for s in engine.evidence_by_expert[eid]]
        scores=engine.resident.selected_scores(query,sources)
        maxima={eid:max((scores[s] for s in engine.evidence_by_expert[eid]),default=0.) for eid in pool}
        lexical=sorted(pool,key=lambda eid:(-maxima[eid],eid))
        retrieval_seconds=time.perf_counter()-start
        start=time.perf_counter();merged=[r['id'] for r in fuse({'candidate':pool,'source_max':lexical},k=20,depth=50,constant=60)];fusion_seconds=time.perf_counter()-start
        # Verify the independently exposed production path, including scores and ownership.
        reference=engine.search_reranked(query,k=20,candidates=50,evidence_per_expert=1)
        if [r['expert']['id'] for r in reference['results']]!=lexical[:20] or any(r['score']!=maxima[r['expert']['id']] for r in reference['results']) or reference['source_snapshot_sha256']!=candidate['source_snapshot_sha256']:
            raise ValueError('Source-max replay differs from production')
        rows.append({'query_id':qid,'retrieval_seconds':retrieval_seconds,'fusion_seconds':fusion_seconds,
                     'rankings':{'candidate_lm':pool[:20],'source_max':lexical[:20],'fusion':merged},
                     'metrics':{t:{a:precision_metrics(rank,labels[t][qid]) for a,rank in {'candidate_lm':pool[:20],'source_max':lexical[:20],'fusion':merged}.items()} for t in labels}})
    engine.evidence.db.close();(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    arms=('candidate_lm','source_max','fusion')
    summary={'queries':len(rows),'matched_production_source_max':len(rows),'initialization_seconds':initialization,
      'mean_common_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),
      'mean_incremental_fusion_seconds':statistics.mean(r['fusion_seconds'] for r in rows),
      'metrics':{t:{a:{m:statistics.mean(r['metrics'][t][a][m] for r in rows) for m in ('p@5','ndcg@10')} for a in arms} for t in labels},
      'fusion_minus_source_max_p5':{t:paired_interval([r['metrics'][t]['fusion']['p@5']-r['metrics'][t]['source_max']['p@5'] for r in rows]) for t in labels},
      'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
      'scope':protocol['scope']+' Shared retrieval time excludes displayed evidence, initialization, model reasoning and index construction. Incremental timing is one observation per query. Unadjusted paired query bootstrap assumes independent queries. Configuration is post-hoc and not held out.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
