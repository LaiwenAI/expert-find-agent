"""Evaluate lexical span proposals and native decisions on public reviewer queries."""
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
from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import FixedPipeline,sha,verify_prior
from experiments.retrieval.cof_structured_candidate_pilot import p3


def main(output,max_queries=20):
    if type(max_queries)!=int or not 1<=max_queries<=20:raise ValueError('Invalid query budget')
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2');hashes=verify_prior(root,papers,reviewers,prior)
    if set(labels['soft'])!=set(labels['hard']) or any(set(labels[t][q])!=set(reviewers) for t in labels for q in labels[t]):raise ValueError('Incomplete judgments')
    structured=Path('artifacts/cof-structured-candidate-pilot-v2');old=json.loads((structured/'protocol.json').read_text());old_summary=json.loads((structured/'summary.json').read_text())
    if sha(structured/'protocol.json')!=old_summary['protocol_sha256'] or sha(structured/'view-hashes.json')!=old_summary['view_hashes_sha256'] or sha(structured/'results.json')!=old_summary['results_sha256']:
        raise ValueError('Changed structured pilot')
    ids=sorted(labels['soft'])[:max_queries]
    if ids!=old['query_ids'][:max_queries] or hashes!=old['input_sha256']:raise ValueError('Changed frozen cohort')
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path('src/expert_search/lexical_anchored_choice_agent.py'),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/progressive_decision_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'input_sha256':hashes,'structured_protocol_sha256':sha(structured/'protocol.json'),'structured_view_hashes_sha256':sha(structured/'view-hashes.json'),
              'code_sha256':{str(p):sha(p) for p in paths},'model':'gpt-6-luna','workers':3,'retries':0,'candidate_pool':50,'candidates':5,'return_k':3,'source_chars':1200,'span_width':400,'anchor_proposal':'local IDF-weighted shared-term overlap',
              'scope':'Reused public reviewer queries and same source views as prior structured pilot. Lexical overlap proposes one span pair per candidate, then one native request judges these pairs. No free-text bridge or independent industrial, source entailment or personal-role labels.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident-adaptive');init=time.perf_counter()-start
    expected=json.loads((structured/'view-hashes.json').read_text());prepared={}
    for qid in ids:
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip();start=time.perf_counter()
        result=deepcopy(engine.search_reranked(query,k=5,candidates=50,evidence_per_expert=1));seconds=time.perf_counter()-start
        for row in result['results']:
            for source in row['evidence']:source['text']=source['text'][:1200]
        view=hashlib.sha256(json.dumps(result['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if view!=expected[qid]:raise ValueError('Changed candidate views')
        ranking=[r['expert']['id'] for r in result['results']]
        if len(set(ranking))!=len(ranking) or set(ranking)-set(reviewers):raise ValueError('Invalid candidate inventory')
        prepared[qid]={'query':query,'result':result,'seconds':seconds,'baseline':ranking[:3]}
    engine.evidence.db.close();(output/'checkpoints').mkdir()
    def run(qid):
        agent=LexicalAnchoredChoiceAgent(FixedPipeline(prepared[qid]['result']),client=LaiwenAI(model='gpt-6-luna',max_retries=0));start=time.perf_counter()
        row={'query_id':qid,'baseline':prepared[qid]['baseline'],'retrieval_seconds':prepared[qid]['seconds'],'status':'failed'}
        try:
            result=agent.run(prepared[qid]['query'],k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200)
            row.update(status=result['status'],unresolved=result.get('unresolved'),decision_seconds=result.get('decision_seconds'))
            if result['status'] in ('complete','partial','empty'):
                row.update(ranking=[x['expert_id'] for x in result['results']],levels=[x['level'] for x in result['results']],
                           anchored=sum(x['source_anchor'] is not None and x['demand_anchor'] is not None for x in result['results']))
            else:row['error_type']=result.get('error_type',result.get('reason'))
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row['model_seconds']=time.perf_counter()-start;row['baseline_p3']={t:p3(row['baseline'],labels[t][qid]) for t in labels}
        if row['status'] in ('complete','partial','empty'):row['model_p3']={t:p3(row['ranking'],labels[t][qid]) for t in labels}
        path=output/'checkpoints'/(hashlib.sha256(qid.encode()).hexdigest()+'.json');temp=path.with_suffix('.partial');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(path)
        print('Completed',row['status'],flush=True);return row
    with ThreadPoolExecutor(max_workers=3) as pool:found={r['query_id']:r for r in (f.result() for f in as_completed([pool.submit(run,q) for q in ids]))}
    rows=[found[q] for q in ids];(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    valid=[r for r in rows if r['status'] in ('complete','partial','empty')];missing=len(rows)-len(valid);metrics={}
    for task in labels:
        values=[r['model_p3'][task] for r in valid];base=[r['baseline_p3'][task] for r in valid]
        metrics[task]={'baseline_all_p3':statistics.mean(r['baseline_p3'][task] for r in rows),'baseline_valid_p3':statistics.mean(base) if base else None,
                       'choice_valid_p3':statistics.mean(values) if values else None,'choice_all_p3_bounds':[sum(values)/len(rows),(sum(values)+missing)/len(rows)],
                       'paired_difference_valid':paired_interval([a-b for a,b in zip(values,base)]) if len(values)>=10 else None}
    report={'planned':len(rows),'valid':len(valid),'statuses':dict(Counter(r['status'] for r in rows)),
            'errors':[{'type':r.get('error_type'),'http_status':r.get('http_status')} for r in rows if r['status'] not in ('complete','partial','empty')],
            'levels':dict(Counter(level for r in valid for level in r['levels'])),'unresolved_candidates':sum(r['unresolved'] for r in valid),
            'returned_anchored':sum(r['anchored'] for r in valid),'mean_returned':statistics.mean(len(r['ranking']) for r in valid) if valid else None,
            'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),'mean_attempt_seconds':statistics.mean(r['model_seconds'] for r in rows),
            'initialization_seconds':init,'metrics':metrics,'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Full model-attempt time excludes shared retrieval and initialization. Choices alone do not provide a transfer explanation. Missing calls are unknown; bounds are not confidence intervals. Bootstrap assumes independent queries, unadjusted.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-queries',type=int,default=20);a=p.parse_args();main(a.output,a.max_queries)
