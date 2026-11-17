"""Compare exact 50-candidate BM25 against certified owner-bound pruning."""
import hashlib
import json
from pathlib import Path
import statistics
import time

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.expert_upper_bound import ExpertTermUpperBounds,certified_candidate_max
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import sha,verify_prior


def main():
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2');hashes=verify_prior(root,papers,reviewers,prior)
    if set(labels['soft'])!=set(labels['hard']) or any(set(labels[t][q])!=set(reviewers) for t in labels for q in labels[t]):raise ValueError('Incomplete labels')
    out=Path('artifacts/cof-certified-candidate-pruning-v2');out.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/expert_upper_bound.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':sorted(labels['soft']),'input_sha256':hashes,'pool':50,'seed':20,'k':5,'evidence_per_expert':1,'arm_order':'alternating',
              'prior_sha256':{str(p.relative_to(prior)):sha(p) for p in (prior/'corpus/experts.jsonl',prior/'corpus/evidence.jsonl',prior/'lm/metadata.json',prior/'lm/arrays.npz',prior/'fast.sqlite')},
              'code_sha256':{str(p):sha(p) for p in code},'scope':'Previously studied fully judged reviewer queries; no independent industrial fit or semantic evidence labels. Exactness is relative to the same 50-candidate max-BM25 path.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident');init=time.perf_counter()-start
    bounds=ExpertTermUpperBounds(engine.resident,engine.evidence_by_expert);rows=[]
    for i,qid in enumerate(protocol['query_ids']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip();attempts={}
        for arm in (('baseline','certified') if i%2==0 else ('certified','baseline')):
            start=time.perf_counter()
            result=(engine.search_reranked(query,k=5,candidates=50,evidence_per_expert=1) if arm=='baseline' else certified_candidate_max(engine,bounds,query))
            attempts[arm]=(result,time.perf_counter()-start)
        baseline,bt=attempts['baseline'];cert,ct=attempts['certified']
        if baseline['results']!=cert['results'] or baseline['source_snapshot_sha256']!=cert['source_snapshot_sha256']:raise ValueError('Certificate disagrees with full scoring')
        ranking=[r['expert']['id'] for r in cert['results']]
        rows.append({'query_id':qid,'baseline_seconds':bt,'certified_seconds':ct,'pool_size':cert['pool_size'],'scored_candidates':cert['scored_candidates'],
                     'pruned_candidates':cert['pruned_candidates'],'ranking':ranking,'metrics':{t:precision_metrics(ranking,labels[t][qid]) for t in labels}})
    engine.evidence.db.close();(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'identical_results':len(rows),'mean_scored_candidates':statistics.mean(r['scored_candidates'] for r in rows),
            'queries_with_pruning':sum(r['pruned_candidates']>0 for r in rows),'mean_pruned_candidates':statistics.mean(r['pruned_candidates'] for r in rows),
            'mean_baseline_seconds':statistics.mean(r['baseline_seconds'] for r in rows),'mean_certified_seconds':statistics.mean(r['certified_seconds'] for r in rows),
            'paired_seconds':paired_interval([r['certified_seconds']-r['baseline_seconds'] for r in rows]),'pipeline_init_seconds':init,
            'bound_init_seconds':bounds.initialization_seconds,'bound_numeric_array_bytes':bounds.array_bytes,
            'mean_p_at_5':{t:statistics.mean(r['metrics'][t]['p@5'] for r in rows) for t in labels},
            'protocol_sha256':sha(out/'protocol.json'),'results_sha256':sha(out/'results.json'),
            'scope':protocol['scope']+' One timing per query, alternating order; initialization, model inference and index construction excluded from query time. Bootstrap assumes independent queries, unadjusted. Numeric-array bytes exclude Python overhead. Exactness is verified only on these queries; sparse bounds depend on the full frozen corpus.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
