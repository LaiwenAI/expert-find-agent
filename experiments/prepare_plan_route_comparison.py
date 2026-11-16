"""Freeze planning variants and silver-evaluable cases before routing tests."""
import hashlib
import json
from pathlib import Path


def run():
    jobs={}
    for path in ('artifacts/paper-backed-overlap-v1/jobs.jsonl','artifacts/expanded-paper-backed-v1/jobs.jsonl'):
        jobs.update({j['id']:j for j in map(json.loads,Path(path).read_text().splitlines())})
    positives={}
    for row in json.loads(Path('artifacts/paper-control-blind-audit-v1/summary.json').read_text())['rows']:
        if row['status']=='dual_model_control_candidate':positives[row['id']]='old_development'
    for row in json.loads(Path('artifacts/expanded-paper-span-audit-v1/summary.json').read_text())['rows']:
        if row['status']=='dual_model_silver_candidate':positives[row['id']]='new_silver'
    keys=json.loads(Path('artifacts/method-bridge-plans-v1/protocol.json').read_text())['queries'];cases=[]
    for key in keys:
        detailed=json.loads((Path('artifacts/method-bridge-plans-v1')/(key+'.json')).read_text())
        compact=json.loads((Path('artifacts/compact-method-plans-reparsed-v2')/(key+'.json')).read_text())
        if detailed['status']!='structurally_valid' or compact['status']!='structurally_valid':raise ValueError('Incomplete planner comparison')
        cases.append(dict(id=key,demand=jobs[key]['demand'],known_expert_id=jobs[key]['expert_id'],evaluation_scope=positives.get(key,'not_evaluable'),
            detailed=detailed['requirements'],compact=compact['requirements'],detailed_planning_seconds=detailed['seconds'],compact_planning_seconds=compact['seconds']))
    out=Path('artifacts/plan-route-comparison-v1');out.mkdir(exist_ok=True);body=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases)
    marker=out/'cases.sha256';digest=hashlib.sha256(body.encode()).hexdigest()
    if marker.exists() and marker.read_text()!=digest:raise ValueError('Changed frozen plan comparison')
    (out/'cases.jsonl').write_text(body);marker.write_text(digest);print('Prepared',len(cases),'cases;',len(positives),'silver-evaluable; others have no relevance label')


if __name__=='__main__':run()
