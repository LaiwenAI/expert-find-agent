"""Paired candidate reasoning strategies on four frozen development needs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior=Path('artifacts/candidate-agent-pilot-v1/protocol.json');old=json.loads(prior.read_text())
    qpath=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(qpath)!=old['query_sha256']:raise ValueError('Changed query source')
    for path,digest in old['inputs_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed source index')
    for name,digest in old['model_files_sha256'].items():
        if sha(Path('artifacts/teacher-compiled-lm')/name)!=digest:raise ValueError('Changed candidate index')
    queries={r['id']:r for r in map(json.loads,qpath.read_text().splitlines()) if r['split']=='development'}
    if len(old['query_ids'])!=4 or any(qid not in queries for qid in old['query_ids']):raise ValueError('Invalid development cohort')
    out=Path('artifacts/enterprise-adaptive-agent-pilot-v1');out.mkdir(parents=True,exist_ok=False)
    protocol=dict(query_ids=old['query_ids'],prior_protocol_sha256=sha(prior),query_sha256=sha(qpath),
                  input_sha256=old['inputs_sha256'],model_files_sha256=old['model_files_sha256'],
                  code_sha256={p:sha(Path(p)) for p in ('src/expert_search/candidate_reasoning_agent.py','src/expert_search/candidate_evidence_pipeline.py','src/expert_search/fast_retrieval.py','experiments/retrieval/enterprise_adaptive_agent_pilot.py')},
                  model='gpt-6-luna',arms=['lm','adaptive_bm25'],order='Alternating arm order; two concurrent query workers, no automatic retries.',
                  scope='Four previously studied development needs. Same one-call exact-quote prompt and one evidence item per candidate, but different candidate sets. Includes retrieval and model calls, excludes client and pipeline initialization. No independent expert-fit or citation-support judgments.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env')
    def run(item):
        i,qid=item;agent=CandidateReasoningAgent(client=LaiwenAI(model='gpt-6-luna',max_retries=0));records={}
        for arm in (protocol['arms'] if i%2==0 else list(reversed(protocol['arms']))):
            start=time.perf_counter();row=dict(query_id=qid,arm=arm,status='unresolved')
            try:
                result=agent.run(queries[qid]['text'],k=5,candidates=20,evidence_per_expert=1,retrieval=arm,rerank_pool=50)
                row.update(status='valid',returned=len(result['results']),rejected=result['rejected_results'],result=result)
            except Exception as error:row['error_type']=type(error).__name__
            row['seconds']=time.perf_counter()-start;records[arm]=row
            (out/(qid+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            print(qid,arm,row['status'],flush=True)
        agent.pipeline.evidence.db.close();return records
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,enumerate(old['query_ids'])))
    report=dict(queries=len(rows),arms={})
    for arm in protocol['arms']:
        values=[r[arm] for r in rows];valid=[r for r in values if r['status']=='valid']
        report['arms'][arm]=dict(valid=len(valid),returned=sum(r['returned'] for r in valid),rejected=sum(r['rejected'] for r in valid),
                                 mean_seconds=sum(r['seconds'] for r in values)/len(values),
                                 prompt_tokens=sum(r['result']['usage']['prompt_tokens'] for r in valid),
                                 completion_tokens=sum(r['result']['usage']['completion_tokens'] for r in valid))
    pairs=[r for r in rows if all(r[a]['status']=='valid' for a in protocol['arms'])]
    report['mean_candidate_overlap']=sum(len(set(r['lm']['result']['candidate_ids'])&set(r['adaptive_bm25']['result']['candidate_ids'])) for r in pairs)/len(pairs) if pairs else None
    report['scope']=protocol['scope'];(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
