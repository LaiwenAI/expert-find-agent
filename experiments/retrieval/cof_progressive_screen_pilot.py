"""Exploratory native Decision screening on reused fully judged CoF SIGIR queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.progressive_decision_agent import ProgressiveDecisionAgent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FixedPipeline:
    def __init__(self, result): self.result = result
    def search_reranked(self, *args, **kwargs): return self.result


def main(output):
    root, papers, reviewers, labels = load('SIGIR')
    if set(labels['soft']) != set(labels['hard']) or any(set(labels[t][q]) != set(reviewers) for t in labels for q in labels[t]):
        raise ValueError('Incomplete judgments')
    prior = Path('artifacts/cof-owned-vote-quality-v2')
    output.mkdir(parents=True, exist_ok=False)
    ids = sorted(labels['soft'])[:20]
    code = [Path(__file__), Path('src/expert_search/progressive_decision_agent.py'), Path('src/expert_search/choice_judgments.py'), Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol = {'dataset':'CoF SIGIR', 'query_selection':'First 20 sorted query IDs, chosen before model calls', 'query_ids':ids,
                'model':'gpt-6-luna', 'automatic_retries':0, 'workers':4, 'rerank_pool':50, 'screen_candidates':20,
                'source_views_per_candidate':1, 'top_k':5, 'explanations_called':False,
                'code_sha256':{str(p):sha(p) for p in code},
                'input_sha256':{n:sha(root/n) for n in ('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')},
                'scope':'Exploratory reused test queries and reviewer-level topic labels, not enterprise expert fit or evidence entailment. Owner-specific source copies alter document frequencies. One model request per query; no retries. Timing excludes initialization and explanation.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    load_dotenv('.env')
    start = time.perf_counter()
    engine = CandidateEvidencePipeline(prior/'corpus', prior/'lm', prior/'fast.sqlite', scoring='resident')
    init = time.perf_counter()-start
    frozen = []
    for qid in ids:
        query = (papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        start = time.perf_counter()
        result = engine.search_reranked(query, k=20, candidates=50, evidence_per_expert=1)
        elapsed = time.perf_counter()-start
        ranked = [r['expert']['id'] for r in result['results']]
        if len(ranked) != len(set(ranked)) or set(ranked)-set(reviewers): raise ValueError('Invalid candidates')
        frozen.append((qid, query, result, elapsed))
    engine.evidence.db.close()
    def run(item):
        qid, query, result, retrieval = item
        agent = ProgressiveDecisionAgent(pipeline=FixedPipeline(result), client=LaiwenAI(model='gpt-6-luna',max_retries=0))
        stream = agent.events(query,k=5,candidates=20,rerank_pool=50,explanation_candidates=20,evidence_per_expert=1)
        first = next(stream); screen = next(stream)
        base = [r['expert']['id'] for r in result['results'][:5]]
        row = {'query_id':qid,'stage':screen['stage'],'retrieval_seconds':retrieval,
               'screen_seconds':screen['seconds']-first['seconds'],'baseline':base,
               'baseline_metrics':{t:precision_metrics(base,labels[t][qid]) for t in labels}}
        if screen['stage']=='screen':
            ranked = [r['expert']['id'] for r in screen['results'][:5]]
            row.update(screened=ranked, selected=screen['selected'], screened_metrics={t:precision_metrics(ranked,labels[t][qid]) for t in labels})
        else: row['error_type']=screen.get('error_type',screen.get('reason'))
        print(qid, row['stage'], flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(run, frozen))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    valid=[r for r in rows if r['stage']=='screen']
    summary={'planned':len(rows),'valid':len(valid),'failures':[{'query_id':r['query_id'],'stage':r['stage'],'error_type':r.get('error_type')} for r in rows if r['stage']!='screen'],
             'initialization_seconds':init, 'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),
             'mean_screen_seconds_valid':statistics.mean(r['screen_seconds'] for r in valid) if valid else None,
             'mean_returned_at_5':statistics.mean(len(r['screened']) for r in valid) if valid else None,
             'metrics':{t:{'baseline_all_p@5':statistics.mean(r['baseline_metrics'][t]['p@5'] for r in rows),
                           'baseline_valid_p@5':statistics.mean(r['baseline_metrics'][t]['p@5'] for r in valid) if valid else None,
                           'screen_valid_p@5':statistics.mean(r['screened_metrics'][t]['p@5'] for r in valid) if valid else None,
                           'screen_all_p@5_bounds':[sum(r['screened_metrics'][t]['p@5'] for r in valid)/len(rows),(sum(r['screened_metrics'][t]['p@5'] for r in valid)+len(rows)-len(valid))/len(rows)]} for t in labels},
             'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
