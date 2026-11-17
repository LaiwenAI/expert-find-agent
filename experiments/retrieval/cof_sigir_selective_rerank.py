"""Candidate shortlist plus selective source scoring on fully judged SIGIR."""
import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch,add_selective_indexes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root,papers,reviewers,labels=load('SIGIR');source=root/'unique-profile-corpus';source_db=source/'fast.sqlite'
    docids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    documents={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in docids}
    profiles={eid:sorted(set(r['papers'])) for eid,r in reviewers.items()}
    index=CompiledCandidateLM(ExpertLanguageModels(documents,profiles));available=set(index.available)
    if available!=set(reviewers):raise ValueError('Missing reviewer profiles')
    qids=sorted(labels['soft'])
    if set(qids)!=set(labels['hard']) or any(set(labels[task][qid])!=available for task in labels for qid in qids):
        raise ValueError('SIGIR candidate judgments incomplete')
    args.output.mkdir(parents=True,exist_ok=False);db=args.output/'selective.sqlite';start=time.perf_counter()
    shutil.copyfile(source_db,db);add_selective_indexes(db);build=time.perf_counter()-start
    search=FastSearch(source,db)
    methods=('candidate_lm','candidate20_bm25','candidate50_bm25','global_bm25');rows=[]
    for i,qid in enumerate(qids):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip();rankings={};seconds={}
        for method in methods[i%4:]+methods[:i%4]:
            start=time.perf_counter()
            if method=='candidate_lm':ranking=[r['id'] for r in certified_topk(index,query,10)['results']]
            elif method=='global_bm25':
                scores=search.document_scores(query)
                values={eid:max((scores.get(pid,0.) for pid in profiles[eid]),default=0.) for eid in available}
                ranking=sorted(available,key=lambda eid:(-values[eid],eid))[:10]
            else:
                n=20 if method=='candidate20_bm25' else 50
                pool=[r['id'] for r in certified_topk(index,query,n)['results']]
                source_ids=sorted({pid for eid in pool for pid in profiles[eid]})
                retrieved=search.evidence_for_selective(query,source_ids,1)
                values={eid:max((retrieved[pid][0]['score'] for pid in profiles[eid] if retrieved.get(pid)),default=0.) for eid in pool}
                ranking=sorted(pool,key=lambda eid:(-values[eid],eid))[:10]
            rankings[method]=ranking;seconds[method]=time.perf_counter()-start
        row=dict(query_id=qid,rankings=rankings,seconds=seconds,metrics={task:{m:precision_metrics(r,labels[task][qid]) for m,r in rankings.items()} for task in labels})
        rows.append(row)
    report=dict(queries=len(rows),reviewers=len(available),additional_index_seconds=build,
                mean_seconds={m:statistics.mean(r['seconds'][m] for r in rows) for m in methods},
                metrics={task:{m:{key:statistics.mean(r['metrics'][task][m][key] for r in rows) for key in ('p@5','ndcg@10')} for m in methods} for task in labels},
                input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')},
                scope='Post-hoc fixed shortlist sizes on fully judged CoF SIGIR reviewer-matching task, not industrial expertise. Single rotating-order query timing excludes initialization, index construction, model reasoning and independent evidence checks. Source label provenance and dependence follow upstream benchmark; no independent confirmation or superiority guarantee.')
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));search.db.close()


if __name__=='__main__':main()
