"""Native Decisions evaluation on frozen blind evidence views; no gold claims."""
import hashlib
import inspect
import json
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend


def run():
    source = Path('artifacts/blind-source-review-v1/inputs.jsonl')
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    out = Path('artifacts/decisions-blind-pilot-v2')
    out.mkdir(exist_ok=True)
    protocol = dict(version=2, backend_sha256=hashlib.sha256(inspect.getsource(DecisionsBackend).encode()).hexdigest(), model='gpt-6-luna', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    unit='One request per evidence view; three complete constraints share the same evidence.')
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed pilot inputs')
    marker.write_text(json.dumps(protocol, indent=2))
    backend = DecisionsBackend()
    results = []
    for row in rows:
        target = out/(row['id']+'.json')
        if target.exists():
            result = json.loads(target.read_text())
        else:
            evidence = json.dumps(row['sources'], ensure_ascii=False)
            pairs = [dict(id=c['id'], constraint=c['text'], text=evidence) for c in row['conditions']]
            try:
                result = dict(id=row['id'], status='valid', **backend.judge(pairs))
            except Exception as error:
                result = dict(id=row['id'], status='error', error_type=type(error).__name__,
                              http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial')
            temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2));temporary.replace(target)
        results.append(result)
        print(row['id'],result['status'],flush=True)
    valid=[r for r in results if r['status']=='valid']
    report=dict(protocol=protocol, views=len(rows),valid_views=len(valid),
        total_request_seconds=sum(r['seconds'] for r in valid),
        dependency_inconsistent_views=sum(next(j['label'] for j in r['judgments'] if j['id']=='relation')=='supported'
            and any(j['label']!='supported' for j in r['judgments'] if j['id'] in ('a','b')) for r in valid),
        human_labels=0,notice='Model judgments only; this pilot does not establish precision or accept dataset labels.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    print(json.dumps(run(),indent=2))
