"""Evaluate deployable global shortlists on fully judged CoF SIGIR reviewers."""
import hashlib
import argparse
import json
import statistics
import time
from pathlib import Path

from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('artifacts/cof-sigir-global-shortlist-v2'));args=parser.parse_args()
    root,papers,reviewers,labels=load('SIGIR')
    ids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    documents={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in ids}
    model=CompiledCandidateLM(ExpertLanguageModels(documents,{eid:r['papers'] for eid,r in reviewers.items()}))
    available=set(model.available)
    if available!=set(reviewers):raise ValueError('Some reviewers lack a usable profile')
    qids=sorted(labels['soft'])
    if set(qids)!=set(labels['hard']):raise ValueError('Label query sets differ')
    if any(set(labels[task][qid])!=available for task in labels for qid in qids):raise ValueError('Query has incompletely judged reviewers')
    lexical=FastSearch(root/'unique-profile-corpus',root/'unique-profile-corpus/fast.sqlite');rows=[]
    for i,qid in enumerate(qids):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip();rankings={};times={}
        for method in (['candidate_lm','bm25_max'] if i%2==0 else ['bm25_max','candidate_lm']):
            started=time.perf_counter()
            if method=='candidate_lm':rankings[method]=[r['id'] for r in certified_topk(model,query,20)['results']]
            else:
                values=lexical.document_scores(query)
                scores={eid:max((values.get(pid,0.) for pid in set(reviewers[eid]['papers'])),default=0.) for eid in available}
                rankings[method]=sorted(scores,key=lambda eid:(-scores[eid],eid))[:20]
            times[method]=time.perf_counter()-started
        metrics={}
        for task in labels:
            positive={eid for eid,score in labels[task][qid].items() if score>0}
            metrics[task]={method:{'p_at_5':sum(eid in positive for eid in rank[:5])/5,
                                   'recall_at_20':sum(eid in positive for eid in rank)/len(positive) if positive else None}
                           for method,rank in rankings.items()}
        rows.append({'query_id':qid,'seconds':times,'metrics':metrics})
    summary={'queries':len(rows),'reviewers':len(available),'mean_seconds':{m:statistics.mean(r['seconds'][m] for r in rows) for m in ('candidate_lm','bm25_max')},
             'metrics':{task:{m:{metric:statistics.mean(r['metrics'][task][m][metric] for r in rows if r['metrics'][task][m][metric] is not None)
                                 for metric in ('p_at_5','recall_at_20')} for m in ('candidate_lm','bm25_max')} for task in labels},
             'source_sha256':hashlib.sha256(json.dumps([documents,reviewers],sort_keys=True).encode()).hexdigest(),
             'input_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')},
             'code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (__file__,'src/expert_search/certified_topk.py','src/expert_search/compiled_language_model.py','src/expert_search/fast_retrieval.py')},
             'scope':'CoF SIGIR paper-to-reviewer task with all reviewers judged, not enterprise matching. One alternating timing per query, excluding index build, initialization, model reasoning and evidence selection. No multiple-comparison adjustment or causal claim.'}
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
