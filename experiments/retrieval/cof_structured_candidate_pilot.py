"""Exploratory structured-method ranking on fully judged public reviewer data."""
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
from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval
from expert_search.structured_candidate_agent import StructuredCandidateAgent
from experiments.retrieval.cof_progressive_screen_pilot import FixedPipeline,sha,verify_prior


def p3(ranking,truth):
    if len(ranking)>3 or len(set(ranking))!=len(ranking) or set(ranking)-set(truth):raise ValueError('Invalid judged ranking')
    return sum(truth[e]>0 for e in ranking)/3


def main(output,max_queries=20):
    if type(max_queries)!=int or not 1<=max_queries<=20:raise ValueError('Invalid query budget')
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2');hashes=verify_prior(root,papers,reviewers,prior)
    if set(labels['soft'])!=set(labels['hard']) or any(set(labels[t][q])!=set(reviewers) for t in labels for q in labels[t]):raise ValueError('Incomplete judgments')
    ids=sorted(labels['soft'])[:max_queries];output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/structured_candidate_agent.py'),Path('src/expert_search/partial_method_match.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'selection':f'First {max_queries} sorted previously studied SIGIR query IDs, fixed before calls','input_sha256':hashes,
              'prior_sha256':{str(p.relative_to(prior)):sha(p) for p in (prior/'corpus/evidence.jsonl',prior/'corpus/experts.jsonl',prior/'lm/metadata.json',prior/'lm/arrays.npz',prior/'fast.sqlite')},
              'code_sha256':{str(p):sha(p) for p in code},'model':'gpt-6-luna','workers':3,'retries':0,'candidate_pool':50,'candidates':5,'return_k':3,'source_chars':1200,
              'scope':'Exploratory reuse of public reviewer labels; model sees one owned source view, whereas labels describe reviewer-level topical relevance. No enterprise fit, individual contribution or source entailment labels. No independent held-out claim.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident-adaptive');init=time.perf_counter()-start;prepared={}
    for qid in ids:
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip();start=time.perf_counter()
        result=engine.search_reranked(query,k=5,candidates=50,evidence_per_expert=1);seconds=time.perf_counter()-start
        result=deepcopy(result)
        for row in result['results']:
            for e in row['evidence']:e['text']=e['text'][:1200]
        ranking=[r['expert']['id'] for r in result['results']]
        if len(set(ranking))!=len(ranking) or set(ranking)-set(reviewers):raise ValueError('Invalid candidate pool')
        prepared[qid]={'query':query,'result':result,'seconds':seconds,'baseline':ranking[:3],
                        'view_sha256':hashlib.sha256(json.dumps(result['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    engine.evidence.db.close();(output/'view-hashes.json').write_text(json.dumps({q:prepared[q]['view_sha256'] for q in ids},indent=2)+'\n');(output/'checkpoints').mkdir()
    def run(qid):
        agent=StructuredCandidateAgent(FixedPipeline(prepared[qid]['result']),client=LaiwenAI(model='gpt-6-luna',max_retries=0));start=time.perf_counter()
        row={'query_id':qid,'baseline':prepared[qid]['baseline'],'retrieval_seconds':prepared[qid]['seconds'],'status':'failed'}
        try:
            result=agent.run(prepared[qid]['query'],k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200)
            row['status']=result['status']
            if result['status']=='structurally_valid':
                row.update(ranking=[x['expert_id'] for x in result['results']],levels=[x['level'] for x in result['results']],completion=result['completion'])
            else:row['error_type']=result.get('matcher',{}).get('error_type',result.get('reason'));row['validation_category']=result.get('matcher',{}).get('validation_category');row['schema_diagnostics']=result.get('matcher',{}).get('schema_diagnostics')
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row['model_seconds']=time.perf_counter()-start
        row['baseline_p3']={t:p3(row['baseline'],labels[t][qid]) for t in labels}
        if row['status']=='structurally_valid':row['model_p3']={t:p3(row['ranking'],labels[t][qid]) for t in labels}
        path=output/'checkpoints'/(hashlib.sha256(qid.encode()).hexdigest()+'.json');tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)
        print('Completed',row['status'],flush=True);return row
    with ThreadPoolExecutor(max_workers=3) as pool:found={r['query_id']:r for r in (f.result() for f in as_completed([pool.submit(run,q) for q in ids]))}
    rows=[found[q] for q in ids];(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    valid=[r for r in rows if r['status']=='structurally_valid'];missing=len(rows)-len(valid)
    metrics={}
    for task in labels:
        values=[r['model_p3'][task] for r in valid];base=[r['baseline_p3'][task] for r in valid]
        metrics[task]={'baseline_all_p3':statistics.mean(r['baseline_p3'][task] for r in rows),
                       'baseline_valid_p3':statistics.mean(base) if base else None,'structured_valid_p3':statistics.mean(values) if values else None,
                       'structured_all_p3_bounds':[sum(values)/len(rows),(sum(values)+missing)/len(rows)],
                       'paired_difference_valid':paired_interval([a-b for a,b in zip(values,base)]) if len(values)>=10 else None}
    report={'planned':len(rows),'valid':len(valid),'statuses':dict(Counter(r['status'] for r in rows)),
            'errors':[{'type':r.get('error_type'),'category':r.get('validation_category'),'schema_diagnostics':r.get('schema_diagnostics'),'http_status':r.get('http_status')} for r in rows if r['status']!='structurally_valid'],
            'completion':dict(Counter(r['completion']['status'] for r in valid)),'levels':dict(Counter(level for r in valid for level in r['levels'])),
            'mean_returned':statistics.mean(len(r['ranking']) for r in valid) if valid else None,
            'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),'mean_attempt_seconds':statistics.mean(r['model_seconds'] for r in rows),
            'initialization_seconds':init,'metrics':metrics,'protocol_sha256':sha(output/'protocol.json'),'view_hashes_sha256':sha(output/'view-hashes.json'),
            'results_sha256':sha(output/'results.json'),'scope':protocol['scope']+' Model and lexical stages not matched in output explanation; timing excludes initialization and shared retrieval from model attempts. Missing calls remain unknown; bounds are not confidence intervals. Bootstrap assumes independent queries and is unadjusted.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-queries',type=int,default=20);a=p.parse_args();main(a.output,a.max_queries)
