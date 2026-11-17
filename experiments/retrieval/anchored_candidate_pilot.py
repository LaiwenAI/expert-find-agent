"""Small local pilot of exact-quote candidate reasoning on historical views."""
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--adaptive',action='store_true');args=parser.parse_args()
    parent=Path('artifacts/local-temporal-author-proxy-v1');summary=json.loads((parent/'summary.json').read_text())
    payload=(parent/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=summary['results_sha256']:raise ValueError('Changed query set')
    queries=[json.loads(x) for x in payload.splitlines()][:4]
    if len(queries)!=4:raise ValueError('Missing pilot queries')
    out=Path('artifacts/anchored-temporal-adaptive-pilot-v1' if args.adaptive else 'artifacts/anchored-temporal-candidate-pilot-v1');out.mkdir(parents=True,exist_ok=False)
    protocol=dict(queries=[q['id'] for q in queries],parent_results_sha256=summary['results_sha256'],retrieval='adaptive_bm25' if args.adaptive else 'lm',
                  scope='Post-hoc four-query author-recovery proxy with unverified known-author labels. One model call per query; exact quote checks validate location and ownership only, not semantic support or personal contribution. Private text and results remain local.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env')
    pipeline=CandidateEvidencePipeline('artifacts/corpus-asof-2020-v1','artifacts/temporal-candidate-lm-2020-v1','artifacts/temporal-evidence-2020-v1.sqlite')
    agent=CandidateReasoningAgent(pipeline=pipeline);rows=[]
    for query in queries:
        start=time.perf_counter();row=dict(query_id=query['id'],status='unresolved')
        try:
            response=agent.run(query['title'],k=5,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25' if args.adaptive else 'lm',rerank_pool=50)
            ids=[r['expert']['id'] for r in response['results']];truth=set(query['known_author_ids'])
            row.update(status='valid',returned=len(ids),hit_at_5=int(bool(set(ids)&truth)),rejected=response['rejected_results'],
                       anchors=sum(len(r['evidence']) for r in response['results']),response=response)
        except Exception as error:row['error_type']=type(error).__name__
        row['seconds']=time.perf_counter()-start;rows.append(row)
        (out/(query['id']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(query['id'],row['status'],flush=True)
    report=dict(queries=len(rows),valid=sum(r['status']=='valid' for r in rows),returned=sum(r.get('returned',0) for r in rows),
                hits=sum(r.get('hit_at_5',0) for r in rows),rejected=sum(r.get('rejected',0) for r in rows),
                anchors=sum(r.get('anchors',0) for r in rows),mean_seconds=sum(r['seconds'] for r in rows)/len(rows),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));pipeline.evidence.db.close()


if __name__=='__main__':main()
