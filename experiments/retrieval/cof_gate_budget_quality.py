"""Post-hoc reviewer-label diagnostic for the live-gate query cohort."""
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
from expert_search.cof_benchmark import load
from expert_search.fast_retrieval import FastSearch
from expert_search.owned_vector_index import ExactOwnedVectorIndex


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root,papers,reviewers,labels=load('SIGIR')
    provenance_path=Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance=json.loads(provenance_path.read_text())
    for path,digest in provenance['input_sha256'].items():
        if sha(path)!=digest: raise ValueError('Changed input: '+path)
    qids=provenance['query_ids'][:30]
    if any(set(labels[t][q])!=set(reviewers) for t in labels for q in qids): raise ValueError('Incomplete labels')
    pids=sorted({p for r in reviewers.values() for p in r['papers']})
    owners={p:[] for p in pids}
    for eid,r in reviewers.items():
        for pid in set(r['papers']): owners[pid].append(eid)
    vectors=np.load(root/'qwen-embedding-torch-v1/vectors.npy',mmap_mode='r',allow_pickle=False)
    if vectors.shape != (len(pids)+len(provenance['query_ids']),1024): raise ValueError('Vector shape mismatch')
    index=ExactOwnedVectorIndex(vectors[:len(pids)],pids,[owners[p] for p in pids])
    lexical=FastSearch(root/'unique-profile-corpus',root/'unique-profile-corpus/fast.sqlite')
    budgets=(1,5,10,20,50); rows=[]
    for qid in qids:
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        ranking=[h['id'] for h in index.search_vector(vectors[len(pids)+provenance['query_ids'].index(qid)],k=max(budgets))['hits']]
        values=lexical.document_scores(query)
        scores={eid:max((values.get(p,0.) for p in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
        ranked=sorted(scores,key=lambda e:(-scores[e],e)); allowed={e for e in ranked[:50] if scores[e]>0}
        results={}
        for budget in budgets:
            dense=ranking[:budget]; gated=[e for e in dense if e in allowed]
            metrics={}
            for task in ('soft','hard'):
                positive={e for e,v in labels[task][qid].items() if v>0}
                metrics[task]={}
                for name,items in [('dense',dense),('gated',gated)]:
                    metrics[task][name]={'recall':len(set(items)&positive)/len(positive) if positive else None,
                        'p_at_5_fixed_denominator':sum(e in positive for e in items[:5])/5,
                        'returned':len(items)}
            results[str(budget)]=metrics
        rows.append({'query_id':qid,'budgets':results})
    summary={'queries':len(rows),'reviewers':len(reviewers),'budgets':{},'input_provenance_sha256':sha(provenance_path),'script_sha256':sha(__file__),
             'scope':'Post-hoc, reused CoF SIGIR queries and complete reviewer self-ratings; cached CPU query vectors, not live GPU timings. Positive means score > 0. Recall divides by all positive reviewers, excluding queries with no positives. P@5 divides by five even when fewer results are returned. The gate filters dense candidates using global lexical top 50 and preserves dense order. No backfill or reranking. Reviewer labels do not establish enterprise fit, evidence support or personal contribution. Cohort and policies are not independent held-out tests.'}
    for budget in budgets:
        summary['budgets'][str(budget)]={}
        for task in ('soft','hard'):
            summary['budgets'][str(budget)][task]={}
            for name in ('dense','gated'):
                entries=[r['budgets'][str(budget)][task][name] for r in rows]
                summary['budgets'][str(budget)][task][name]={k:statistics.mean(x[k] for x in entries if x[k] is not None) for k in ('recall','p_at_5_fixed_denominator','returned')}
                summary['budgets'][str(budget)][task][name]['queries_with_positive']=sum(x['recall'] is not None for x in entries)
    out=Path('artifacts/cof-gate-budget-quality-v1');out.mkdir(parents=True,exist_ok=False)
    (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');summary['rows_sha256']=sha(out/'rows.json')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
