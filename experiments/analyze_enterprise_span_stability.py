"""Descriptive paired-case comparison, retaining missing-response counts."""
import json
from pathlib import Path
import numpy as np


def run():
    first=Path('artifacts/enterprise-span-assessment-v1')
    second=Path('artifacts/enterprise-span-repeat-v1')
    a=json.loads((first/'evaluation.json').read_text());b=json.loads((second/'evaluation.json').read_text())
    pairs={}
    for run,rows in [('first',a),('repeat',b)]:
        for r in rows:pairs.setdefault((r['id'],r['method']),{})[run]=r
    methods=sorted({r['method'] for r in a});views=sorted({r['id'] for r in a})
    complete=[v for v in views if all(set(pairs.get((v,m),{}))=={'first','repeat'} and all(r['outcome']=='structurally_valid' for r in pairs[(v,m)].values()) for m in methods)]
    report=dict(total_views=len(views),both_runs_all_methods_valid_views=len(complete),
                descriptive_common_view_means={m:{run:float(np.mean([pairs[(v,m)][run]['coverage'] for v in complete])) for run in ['first','repeat']} for m in methods},
                first_unique_request_count=json.loads((first/'summary.json').read_text())['unique_requests'],
                repeat_analysis=json.loads((second/'repeat-analysis.json').read_text())['summary'],
                warning='Complete-case descriptive model scores only, alongside all failure counts. Same5expert-demand pairs, no independent human labels. Does not establish accuracy, uncertainty coverage or significance; timeout has missing duration, no latency comparison.')
    (second/'stability-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
