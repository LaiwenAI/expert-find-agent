"""Conjunction diagnostic; symbolic reference exploits an explicit query grammar."""
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from .benchmark import metrics
from .corpus import records
from .fast_retrieval import FastSearch


def run(output='artifacts/controlled-constraints/results'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    index=defaultdict(set)
    for _,r in records('artifacts/corpus/evidence.jsonl'):
        if r.get('field')=='\u7814\u7a76\u9886\u57df_CN':
            for line in r['text'].splitlines():index[line.strip()].add(r['expert_id'])
    engine=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite');rows=[]
    for _,q in records('artifacts/controlled-constraints/queries.jsonl'):
        for method in ['bm25-max','bm25-vote','explicit-conjunction-reference']:
            target=out/f"{q['id']}-{method}.json"
            if target.exists():row=json.loads(target.read_text())
            else:
                start=time.perf_counter()
                if method=='explicit-conjunction-reference':
                    terms=re.findall('\u300c([^\u300d]+)\u300d',q['query'])
                    if len(terms)!=2:raise ValueError('Unsupported query grammar')
                    ranking=sorted(index[terms[0]]&index[terms[1]])[:10]
                else:ranking=[r['id'] for r in engine.search(q['query'],10,method)]
                elapsed=time.perf_counter()-start
                row=dict(query_id=q['id'],method=method,ranking=ranking,metrics=metrics(ranking,{x:1 for x in q['known_both']}),seconds=elapsed,
                    returned_one_condition=sum(x in q['known_one'] for x in ranking))
                target.write_text(json.dumps(row,indent=2))
            rows.append(row)
    summary={}
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method];n=len(rs)
        summary[method]=dict(queries=n,metrics={key:sum(r['metrics'][key] for r in rs)/n for key in rs[0]['metrics']},
            mean_seconds=sum(r['seconds'] for r in rs)/n,one_condition_results=sum(r['returned_one_condition'] for r in rs))
    (out/'summary.json').write_text(json.dumps(dict(results=summary,warning='Exact research-area phrase conjunction, synthetic grammar. Symbolic reference is NOT an industrial semantic expert-search performance claim.'),indent=2));return summary

if __name__=='__main__':print(json.dumps(run(),indent=2))
