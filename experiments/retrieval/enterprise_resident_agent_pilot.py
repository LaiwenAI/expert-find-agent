"""Compare full agent calls with identical SQLite and resident retrieval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(output):
    source = Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    prior = Path('artifacts/enterprise-adaptive-agent-pilot-v1/protocol.json'); previous = json.loads(prior.read_text())
    if sha(source) != previous['query_sha256']: raise ValueError('Changed source needs')
    queries = {r['id']: r for r in map(json.loads, source.read_text().splitlines()) if r['split'] == 'development'}
    ids = previous['query_ids']
    if len(ids) != 4 or len(set(ids)) != 4 or set(ids) - set(queries): raise ValueError('Changed pilot cohort')
    output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/fast_retrieval.py'),Path('src/expert_search/agent.py')]
    files = [Path('artifacts/corpus/experts.jsonl'),Path('artifacts/corpus/evidence.jsonl'),Path('artifacts/fast-search-selective-v1.sqlite'),Path('artifacts/teacher-compiled-lm/arrays.npz'),Path('artifacts/teacher-compiled-lm/metadata.json')]
    protocol = {'query_ids': ids, 'query_sha256': sha(source), 'prior_protocol_sha256': sha(prior), 'input_sha256': {str(p): sha(p) for p in files}, 'code_sha256': {str(p): sha(p) for p in code},
                'model': 'gpt-6-luna', 'arms': ['sqlite','resident'], 'workers': 2, 'automatic_retries': 0,
                'scope': 'Four previously studied development needs; identical candidate50/return20 retrieval and one visible evidence item per candidate. Alternating method order, two concurrent workers with separate initialized pipelines. One native chat request per arm, exact prompt hash compared; model output may vary. Includes retrieval and model calls but excludes initialization and index construction; failures retained. No independent expert-fit or evidence-entailment labels.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n'); load_dotenv('.env')
    def worker(indexed):
        i, qid = indexed; start = time.perf_counter(); engine = CandidateEvidencePipeline(scoring='resident'); engine.evidence.prepare_adaptive_scoring()
        initialization = time.perf_counter() - start; resident = engine.resident
        agent = CandidateReasoningAgent(pipeline=engine,client=LaiwenAI(model='gpt-6-luna',max_retries=0)); original_ask = agent.helper.ask
        capture = {}
        def ask(stage, instruction, payload):
            capture['prompt_sha256'] = hashlib.sha256(json.dumps({'stage':stage,'instruction':instruction,'payload':payload},ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return original_ask(stage,instruction,payload)
        agent.helper.ask = ask; attempts = {}
        for arm in (protocol['arms'] if i % 2 == 0 else list(reversed(protocol['arms']))):
            capture.clear(); engine.resident = None if arm == 'sqlite' else resident
            start = time.perf_counter(); row = {'query_id':qid,'arm':arm,'status':'failed'}
            try:
                result = agent.run(queries[qid]['text'], k=5,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
                row.update(status='valid', returned=len(result['results']), rejected=result['rejected_results'],
                           ranking=[r['expert']['id'] for r in result['results']], usage=result['usage'], retrieval_seconds=result['timing']['retrieval']['total_seconds'])
            except Exception as error:
                row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            row.update(seconds=time.perf_counter()-start,prompt_sha256=capture.get('prompt_sha256')); attempts[arm]=row
            (output / (qid+'-'+arm+'.json')).write_text(json.dumps(row,indent=2)+'\n'); print('Completed query',i+1,arm,row['status'],flush=True)
        engine.evidence.db.close()
        return {'attempts':attempts,'initialization_seconds':initialization,'resident_build_seconds':resident.initialization_seconds,'resident_array_bytes':resident.array_bytes}
    with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(worker,enumerate(ids)))
    metrics = {}
    for arm in protocol['arms']:
        items=[r['attempts'][arm] for r in results]; valid=[x for x in items if x['status']=='valid']
        metrics[arm]={'attempts':len(items),'valid':len(valid),'mean_attempt_seconds':sum(x['seconds'] for x in items)/len(items),
                      'mean_retrieval_seconds_valid':sum(x['retrieval_seconds'] for x in valid)/len(valid) if valid else None,
                      'returned':sum(x['returned'] for x in valid),'rejected':sum(x['rejected'] for x in valid),
                      'prompt_tokens_valid':sum(x['usage']['prompt_tokens'] for x in valid), 'completion_tokens_valid':sum(x['usage']['completion_tokens'] for x in valid)}
    pairs=[r['attempts'] for r in results if all(r['attempts'][a]['status']=='valid' for a in protocol['arms'])]
    report={'queries':len(results),'arms':metrics,'valid_pairs':len(pairs),'matching_prompt_pairs':sum(p['sqlite']['prompt_sha256']==p['resident']['prompt_sha256'] for p in pairs),
            'same_ordered_results':sum(p['sqlite']['ranking']==p['resident']['ranking'] for p in pairs),
            'worker_initialization_seconds':[r['initialization_seconds'] for r in results], 'resident_build_seconds':[r['resident_build_seconds'] for r in results],
            'resident_array_bytes':results[0]['resident_array_bytes'],'protocol_sha256':sha(output/'protocol.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); main(p.parse_args().output)
