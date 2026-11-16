"""Frozen Chinese concept-index source recovery; incomplete labels, no training."""
import json
from pathlib import Path
import numpy as np
from .chinese_concept_index import ConceptSearch
from .benchmark import metrics
from .progressive_dense import top_indices


def run(output='artifacts/chinese-concept-diagnostic'):
    engine=ConceptSearch();out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(Path('artifacts/semantic-constraints').glob('conjunction-*.json')):
        r=json.loads(p.read_text())
        if r['status']!='audited_silver':continue
        target=out/p.name
        if target.exists():row=json.loads(target.read_text())
        else:
            result=engine.search(r['query']);ranking=[x['id'] for x in result['results']]
            row=dict(query_id=r['id'],ranking=ranking,known_source_metrics=metrics(ranking,{x:1 for x in r['known_source_experts']}),seconds=result['seconds'],evidence=result['results'])
            target.write_text(json.dumps(row,ensure_ascii=False,indent=2))
        rows.append(row)
    summary=dict(queries=len(rows),metrics={key:float(np.mean([r['known_source_metrics'][key] for r in rows])) for key in rows[0]['known_source_metrics']},
        p50_seconds=float(np.median([r['seconds'] for r in rows])),p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),
        warning='Known-source recovery only; model-audited paraphrases and incomplete labels. Does not measure real enterprise matching accuracy.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':print(json.dumps(run(),indent=2))
