"""Lexical leakage and known-source recovery diagnostics, never relevance gold."""
import json
import time
from pathlib import Path
from .retrieval import tokens
from .fast_retrieval import FastSearch


def run(source='artifacts/synthesis-evidence-first',audit='artifacts/synthesis-evidence-first-audit',output='docs/exploration/synthetic-diagnostics.json'):
    engine=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite');rows=[]
    for path in sorted(Path(source).glob('*.json')):
        r=json.loads(path.read_text());query=r['example'].get('query','')
        if r['status']=='rejected' or not query:continue
        a=Path(audit)/path.name
        verification=json.loads(a.read_text())['status'] if a.exists() else 'pending'
        q=set(tokens(query));source_terms=set(tokens(' '.join(x['text'] for x in r['sources'].values())))
        methods={}
        for method in ['bm25-max','bm25-vote','tfidf-max']:
            start=time.perf_counter();result=engine.search(query,10,method)
            rank=next((i for i,x in enumerate(result,1) if x['id']==r['expert_id']),None)
            methods[method]=dict(source_rank_at_10=rank,seconds=time.perf_counter()-start)
        rows.append(dict(id=path.stem,verification=verification,query_term_coverage_by_source=len(q&source_terms)/len(q) if q else 0,
                         methods=methods))
    report=dict(task='Known source recovery and lexical overlap; not complete expert relevance',rows=rows,
        warning='Source evidence remains indexed. High recovery can reflect construction artifacts. No performance comparison against original demand is valid.')
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2));return report
