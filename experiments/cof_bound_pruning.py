"""Oracle-cost feasibility of per-term bounds for exact BM25-top3 ranking.

All document scores are still computed. This measures avoided expert aggregation,
not avoided posting work, and is related to standard upper-bound retrieval.
"""
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load
from expert_search.retrieval import tokens


def run():
    root,papers,reviewers,labels=load('NIPS');database=root/'unique-profile-corpus/fast.sqlite'
    owners=defaultdict(list)
    for eid,r in reviewers.items():
        for pid in set(r['papers']):owners[pid].append(eid)
    connection=sqlite3.connect(database)
    documents={doc:pid for doc,pid in connection.execute('SELECT id,evidence_id FROM docs')}
    top=defaultdict(lambda:defaultdict(list))
    for term,doc,value in connection.execute('SELECT term,doc,bm25 FROM scores'):
        for eid in owners[documents[doc]]:
            values=top[term][eid];values.append(value)
            if len(values)>3:values.remove(min(values))
    connection.close()
    # Upward rounded additions/division bound top3 aggregation, with extra
    # accumulation inflation for per-document query-term rounding.
    bounds={term:{eid:float(np.nextafter(math.fsum(values)/3,np.inf)) for eid,values in group.items()} for term,group in top.items()}
    engine=ArrayBM25(database);rows=[]
    for qid in sorted(labels['soft']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip();terms=sorted(set(tokens(query)))
        upper={eid:0. for eid in labels['soft'][qid]}
        for term in terms:
            for eid,value in bounds.get(term,{}).items():
                if eid in upper:upper[eid]=float(np.nextafter(upper[eid]+value,np.inf))
        # Conservative standard gamma_n relative error for positive sums.
        eps=np.finfo(np.float64).eps;n=len(terms)+4
        inflation=(1+n*eps)/(1-n*eps)
        upper={eid:float(np.nextafter(value*inflation,np.inf)) for eid,value in upper.items()}
        doc_scores=engine.document_scores(query)
        def exact(eid):return sum(sorted((doc_scores.get(pid,0.) for pid in set(reviewers[eid]['papers'])),reverse=True)[:3])/3
        reference={eid:exact(eid) for eid in upper}
        if any(reference[eid]>upper[eid] for eid in upper):raise AssertionError('Bound violated')
        order=sorted(upper,key=lambda eid:(-upper[eid],eid));computed={}
        for eid in order:
            ranked=sorted(computed,key=lambda e:(-computed[e],e))
            if len(ranked)>=5 and (upper[eid]<computed[ranked[4]] or (upper[eid]==computed[ranked[4]] and eid>ranked[4])):break
            computed[eid]=exact(eid)
        ranking=sorted(computed,key=lambda e:(-computed[e],e))[:5]
        expected=sorted(reference,key=lambda e:(-reference[e],e))[:5]
        if ranking!=expected:raise AssertionError('Changed ordered top5')
        rows.append(dict(query_id=qid,eligible=len(upper),evaluated=len(computed),exact_top5=True))
    report=dict(dataset='NIPS',queries=len(rows),expert_aggregations=sum(r['evaluated'] for r in rows),full_aggregations=sum(r['eligible'] for r in rows),rows=rows,
        scope='Feasibility replay: all document/posting scores computed; only expert aggregation potentially skipped. Numerical bound checked on all candidates, not a general floating-point proof. No live speed claim.')
    Path('artifacts/cof-bound-pruning-nips.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':run()
