"""Known-source recovery for audited paraphrases, with incomplete relevance labels."""
import json
from pathlib import Path
from .fast_retrieval import FastSearch
from .benchmark import metrics


def run(output='artifacts/semantic-constraints/retrieval-diagnostic.json'):
    engine=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite');rows=[];attempts=0
    for p in sorted(Path('artifacts/semantic-constraints').glob('conjunction-*.json')):
        r=json.loads(p.read_text());attempts+=1
        if r['status']!='audited_silver':continue
        for method in ['bm25-max','bm25-vote','tfidf-max']:
            result=engine.search(r['query'],10,method);ranking=[x['id'] for x in result]
            rows.append(dict(query_id=r['id'],method=method,ranking=ranking,
                known_source_metrics=metrics(ranking,{eid:1 for eid in r['known_source_experts']})))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method]
        summary[method]={key:sum(r['known_source_metrics'][key] for r in selected)/len(selected) for key in selected[0]['known_source_metrics']}
    report=dict(attempts_available=attempts,audited_queries=len(rows)//3,results=summary,rows=rows,
        warning='Known-source recovery, incomplete semantic qrels. Unlabeled experts may be relevant. Not precision or human expert-search accuracy.')
    Path(output).write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    r=run();print({k:v for k,v in r.items() if k!='rows'})
