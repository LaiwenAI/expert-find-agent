import json
from pathlib import Path
import numpy as np
from expert_search.area_retrieval import AreaSearch
from expert_search.benchmark import metrics
index=AreaSearch();rows=[]
for p in sorted(Path('artifacts/semantic-constraints').glob('conjunction-*.json')):
 q=json.loads(p.read_text())
 if q['status']!='audited_silver':continue
 for method in ['bm25','coverage']:
  r=index.search(q['query'],method=method)
  rows.append(dict(id=q['id'],method=method,seconds=r['seconds'],metrics=metrics([x['id'] for x in r['results']],{x:1 for x in q['known_source_experts']})))
summary={m:dict(queries=len([r for r in rows if r['method']==m]),metrics={key:float(np.mean([r['metrics'][key] for r in rows if r['method']==m])) for key in rows[0]['metrics']},p50_seconds=float(np.median([r['seconds'] for r in rows if r['method']==m]))) for m in ['bm25','coverage']}
Path('docs/exploration/area-diagnostic.json').write_text(json.dumps({'results':summary,'warning':'Known-source recovery on model-audited easy paraphrases; compact area-only baselines, not complete industrial tasks or proof of novelty.'},indent=2));print(summary)
