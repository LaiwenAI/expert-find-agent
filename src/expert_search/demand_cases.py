"""Run qualitative industrial cases with all lexical baselines and model traces."""
import json
import time
from pathlib import Path
from .agent import ExpertAgent
from .corpus import records


def run(queries='artifacts/demands/queries.jsonl', output='artifacts/demand-cases', limit=3):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    agent=ExpertAgent()
    for _,q in list(records(queries))[:limit]:
        target=out/f"{q['id']}.json"
        if target.exists(): continue
        start=time.perf_counter(); baselines={}
        for method in ['bm25-max','bm25-vote','tfidf-max']:
            baselines[method]=agent.searcher.search(q['query'],5,method)
        try:
            response=agent.run(q['query'],5,10)
            result=dict(status='ok',agent=response)
        except Exception as e:
            result=dict(status='error',error=f'{type(e).__name__}: {e}')
        target.write_text(json.dumps(dict(query=q,baselines=baselines,**result,seconds=time.perf_counter()-start),ensure_ascii=False,indent=2))
    return str(out)
