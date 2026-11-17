"""Export real demand queries and a blinded relevance annotation pool."""
import json
import random
from pathlib import Path
from .retrieval import Search


def export(data='data', output='artifacts/demands', n=20, seed=42):
    from openpyxl import load_workbook
    source=next((Path(data)/'\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42').glob('*.xlsx'))
    book=load_workbook(source,read_only=True,data_only=True)
    sheet=book.active
    rows=list(sheet.values); header=list(rows[0]); records=[]
    for line,values in enumerate(rows[1:],2):
        r=dict(zip(header,values))
        if r.get('\u547d\u9898\u540d\u79f0') and r.get('\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42'):
            records.append(dict(id=f'demand-{line}',title=r['\u547d\u9898\u540d\u79f0'],query=r['\u547d\u9898\u540d\u79f0']+'\n'+r['\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42'],url=r.get('\u8be6\u60c5\u94fe\u63a5'),source=f'{source}:{sheet.title}:{line}'))
    random.Random(seed).shuffle(records); records=records[:n]
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    (out/'queries.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n')
    return records


def pool(queries, output='artifacts/demands', corpus='artifacts/corpus', database='artifacts/search.sqlite'):
    engine=Search(corpus,database); runs=[]; annotations=[]
    for q in queries:
        candidates={}
        for method in ['bm25-max','bm25-vote','tfidf-max']:
            result=engine.search(q['query'],10,method)
            runs.append(dict(query_id=q['id'],method=method,ranking=[x['id'] for x in result]))
            for r in result: candidates.setdefault(r['id'],r)
        for eid in sorted(candidates):
            r=candidates[eid]
            annotations.append(dict(query_id=q['id'],expert_id=eid,name=r.get('name'),query=q['query'],
                evidence=r['evidence'],grade=None,annotator=None,rationale=''))
    out=Path(output)
    for name,rows in [('baseline-runs',runs),('annotation-pool',annotations)]:
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    return dict(queries=len(queries),annotation_pairs=len(annotations))
