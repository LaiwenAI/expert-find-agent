"""Resume model evaluation against the exact lexical benchmark queries and exclusions."""
import json
import hashlib
import time
from pathlib import Path
from .agent import ExpertAgent
from .benchmark import metrics


def run(benchmark='artifacts/benchmark', output='artifacts/agent-benchmark', limit=10, compiled_database=None):
    root=Path(benchmark); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    signature=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    marker=out/'input-fingerprint.txt'
    if marker.exists() and marker.read_text()!=signature:
        raise ValueError('Benchmark input changed; choose a fresh output directory')
    marker.write_text(signature)
    code_marker=out/'agent-code-fingerprint.txt'
    code_files=('agent.py','agent_benchmark.py','retrieval.py','fast_retrieval.py','expert_catalog.py','corpus.py','benchmark.py')
    code_hash=hashlib.sha256()
    for name in code_files:
        payload=Path(__file__).with_name(name).read_bytes()
        code_hash.update(name.encode()+b'\0'+str(len(payload)).encode()+b'\0'+payload)
    code_signature=code_hash.hexdigest()
    config_marker=out/'retrieval-configuration.json'
    config={'compiled_database':str(Path(compiled_database).resolve()) if compiled_database else None}
    if compiled_database:
        h=hashlib.sha256()
        with Path(compiled_database).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        config['compiled_sha256']=h.hexdigest()
    config_text=json.dumps(config,sort_keys=True)
    if config_marker.exists() and config_marker.read_text()!=config_text:
        raise ValueError('Retrieval configuration changed; choose a fresh output directory')
    if code_marker.exists() and code_marker.read_text()!=code_signature:
        raise ValueError('Agent benchmark code changed; choose a fresh output directory')
    if not code_marker.exists() and any(out.glob('*.json')):
        raise ValueError('Existing results have no code fingerprint; choose a fresh output directory')
    code_marker.write_text(code_signature)
    config_marker.write_text(config_text)
    queries={}
    for line in (root/'runs.jsonl').read_text().splitlines():
        r=json.loads(line); queries.setdefault(r['query_id'],r)
    agent=ExpertAgent(compiled_database=compiled_database); excluded=set(manifest['excluded_evidence'])
    for qid,q in list(queries.items())[:limit]:
        path=out/f'{qid}.json'
        if path.exists(): continue
        start=time.perf_counter()
        try:
            response=agent.run(q['query'],k=10,candidates=20,exclude=excluded)
            ranking=[r['expert']['id'] for r in response['results']]
            row=dict(query_id=qid,query=q['query'],qrels=q['qrels'],method='hybrid-evidence-tree',
                ranking=ranking,metrics=metrics(ranking,q['qrels']),response=response,status='ok')
        except Exception as e:
            row=dict(query_id=qid,status='error',error=f'{type(e).__name__}: {e}',metrics=metrics([],q['qrels']))
        row['seconds']=time.perf_counter()-start
        path.write_text(json.dumps(row,ensure_ascii=False,indent=2))
    rows=[json.loads((out/f'{qid}.json').read_text()) for qid in list(queries)[:limit]]
    summary=dict(queries=len(rows),failures=sum(r['status']!='ok' for r in rows),
        metrics={key:sum(r['metrics'][key] for r in rows)/len(rows) for key in rows[0]['metrics']},
        mean_seconds=sum(r['seconds'] for r in rows)/len(rows),benchmark_manifest=manifest)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
