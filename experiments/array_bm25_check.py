"""Paired exact-score check against the existing SQLite document scorer."""
import json
import random
import time
from pathlib import Path
from expert_search.array_bm25 import ArrayBM25
from expert_search.fast_retrieval import FastSearch
from expert_search.cof_benchmark import load


def run():
    root,papers,_,labels=load('NIPS');corpus=root/'unique-profile-corpus'
    old=FastSearch(corpus,corpus/'fast.sqlite')
    start=time.perf_counter();new=ArrayBM25(corpus/'fast.sqlite');initialization=time.perf_counter()-start
    rows=[]
    for qid in sorted(labels['soft']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        order=['sqlite','array'];random.Random(qid).shuffle(order);results={};seconds={}
        for name in order:
            start=time.perf_counter();results[name]=(old if name=='sqlite' else new).document_scores(query);seconds[name]=time.perf_counter()-start
        if results['sqlite']!=results['array']:raise AssertionError('Changed BM25 document scores')
        rows.append(dict(query_id=qid,seconds=seconds,exact=True))
    report=dict(dataset='NIPS',queries=len(rows),exact_all=True,initialization_seconds=initialization,array_bytes=new.array_bytes,
        total_seconds={name:sum(r['seconds'][name] for r in rows) for name in ('sqlite','array')},rows=rows,
        scope='Resident engineering baseline; scoring includes materializing document-score dictionaries. One repeat per query, no ranking-quality change.')
    out=Path('artifacts/array-bm25-check.json');out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':run()
