"""Private paired-view pilot of lexical proposals and native choice judgments."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.lexical_anchored_choice_agent import LexicalAnchoredChoiceAgent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenPipeline:
    def __init__(self,result):self.result=result
    def search_reranked(self,*args,**kwargs):return self.result


def main(output):
    prior=Path('artifacts/enterprise-structured-candidate-pilot-v1');p=json.loads((prior/'protocol.json').read_text());s=json.loads((prior/'summary.json').read_text())
    if sha(prior/'protocol.json')!=s['protocol_sha256'] or sha(prior/'view-hashes.json')!=s['view_hashes_sha256']:raise ValueError('Changed structured pilot')
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=p['queries_sha256']:raise ValueError('Changed query source')
    ids=p['query_ids'];queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'}
    if len(ids)!=4 or len(set(ids))!=4 or any(q not in queries for q in ids):raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False);paths=[Path(__file__),Path('src/expert_search/lexical_anchored_choice_agent.py'),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'queries_sha256':sha(source),'prior_protocol_sha256':sha(prior/'protocol.json'),'prior_views_sha256':sha(prior/'view-hashes.json'),
              'code_sha256':{str(f):sha(f) for f in paths},'model':'gpt-6-luna','workers':2,'retries':0,'candidates':5,'return_k':3,'candidate_pool':50,
              'scope':'Four reused private development needs and previously frozen candidate/source prefixes. Lexical span proposals and one native choice call do not generate an explanatory bridge. No independent relevance, source entailment or personal-role labels.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    engine=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start;expected=json.loads((prior/'view-hashes.json').read_text());prepared={};previous={}
    for qid in ids:
        old_file=prior/(hashlib.sha256(qid.encode()).hexdigest()+'.json');old=json.loads(old_file.read_text())
        if old['query_id']!=qid:raise ValueError('Changed prior result')
        previous[qid]=old['arms']['structured']
        start=time.perf_counter();result=deepcopy(engine.search_reranked(queries[qid]['text'],k=5,candidates=50,evidence_per_expert=1));elapsed=time.perf_counter()-start
        for row in result['results']:
            for source in row['evidence']:source['text']=source['text'][:1200]
        fingerprint=hashlib.sha256(json.dumps(result['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if fingerprint!=expected[qid]:raise ValueError('Changed candidate/source view')
        prepared[qid]={'result':result,'seconds':elapsed}
    engine.evidence.db.close()
    def run(qid):
        agent=LexicalAnchoredChoiceAgent(FrozenPipeline(prepared[qid]['result']),client=LaiwenAI(model='gpt-6-luna',max_retries=0));start=time.perf_counter()
        attempt={'query_id':qid,'status':'failed'}
        try:
            result=agent.run(queries[qid]['text'],k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200)
            attempt.update(status=result['status'],result=result)
        except Exception as error:attempt.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        attempt['seconds']=time.perf_counter()-start;path=output/(hashlib.sha256(qid.encode()).hexdigest()+'.json');tmp=path.with_suffix('.partial')
        tmp.write_text(json.dumps(attempt,ensure_ascii=False,indent=2)+'\n');tmp.replace(path);print('Completed',attempt['status'],flush=True);return attempt
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[f.result() for f in as_completed([pool.submit(run,qid) for qid in ids])]
    valid=[r for r in rows if r['status'] in ('complete','partial','empty')];pairs=[r for r in valid if previous[r['query_id']]['status']=='valid']
    levels=Counter(x['level'] for r in valid for x in r['result']['results'])
    report={'attempts':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),'levels':dict(levels),'model_positive_returned':sum(r['result']['model_positive_returned'] for r in valid),
            'unresolved_candidates':sum(r['result']['unresolved'] for r in valid),'returned_candidates':sum(r['result']['returned'] for r in valid),
            'prior_structured_valid_pairs':len(pairs),'same_ordered_rankings':sum([x['expert_id'] for x in r['result']['results']]==previous[r['query_id']]['ranking'] for r in pairs),
            'mean_top_three_overlap':statistics.mean(len({x['expert_id'] for x in r['result']['results']}&set(previous[r['query_id']]['ranking']))/3 for r in pairs) if pairs else None,
            'mean_attempt_seconds':statistics.mean(r['seconds'] for r in rows),'mean_shared_retrieval_seconds':statistics.mean(v['seconds'] for v in prepared.values()),
            'pipeline_init_seconds':init,'protocol_sha256':sha(output/'protocol.json'),
            'scope':protocol['scope']+' Prior structured calls occurred earlier with different prompt and output schema; timing and rank differences do not establish a causal quality or speed advantage. Fixed denominator three includes missing outputs.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
