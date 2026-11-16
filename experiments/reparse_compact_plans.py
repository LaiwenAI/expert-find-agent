"""Reparse saved responses by character budget; not another latency run."""
import hashlib
import json
from pathlib import Path
from compact_method_plans import parse_paths
from audit_paper_backed_spans import spans


def run():
    parent=Path('artifacts/compact-method-plans-v1');out=Path('artifacts/compact-method-plans-reparsed-v2');out.mkdir(exist_ok=True)
    jobs={}
    for path in ('artifacts/paper-backed-overlap-v1/jobs.jsonl','artifacts/expanded-paper-backed-v1/jobs.jsonl'):
        jobs.update({j['id']:j for j in map(json.loads,Path(path).read_text().splitlines())})
    spec=json.loads((parent/'protocol.json').read_text());protocol=dict(version=2,parent_protocol=spec,policy='Posthoc parser variant: retain original 180-character query limit, omit redundant word-count limit. Same raw model responses and observed timings; no API rerun or new semantic validation.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed reparse protocol')
    marker.write_text(json.dumps(protocol,indent=2));rows=[]
    for key in spec['queries']:
        path=parent/(key+'.json');row=json.loads(path.read_text());row['parent_response_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            answer=json.loads(row['response']['choices'][0]['message']['content']);row['requirements']=parse_paths(answer['paths'],spans(jobs[key]['demand']),max_words=None);row['status']='structurally_valid'
        except Exception as error:row.update(status='unresolved',error_type=type(error).__name__)
        (out/path.name).write_text(json.dumps(row,ensure_ascii=False,indent=2));rows.append(row)
    report=dict(queries=len(rows),valid=sum(r['status']=='structurally_valid' for r in rows),inherited_observed_seconds=sum(r['seconds'] for r in rows),
        warning=protocol['policy']);(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
