"""Reuse the identical arm-blind witness transport on a new fixed cohort."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import audit_focused_witnesses as transport

DATA=Path('artifacts/concrete-partial-comparison-v1');ROOT=Path('artifacts/concrete-witness-audit-v1')


def run():
    protocol=json.loads((DATA/'protocol.json').read_text());paths=[DATA/(q+'.json') for q in protocol['query_ids']];rows=[json.loads(p.read_text()) for p in paths]
    jobs_path=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs={j['id']:j for j in map(json.loads,jobs_path.read_text().splitlines()) if j['split']=='development'}
    if any(r['query_id'] not in jobs for r in rows):raise ValueError('Not development')
    ROOT.mkdir(exist_ok=True);transport.ROOT=ROOT
    frozen=dict(model='gpt-6-luna',inputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},jobs_sha256=hashlib.sha256(jobs_path.read_bytes()).hexdigest(),code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/audit_focused_witnesses.py'),Path('experiments/annotate_fresh_evaluation.py'),Path('src/expert_search/choice_judgments.py')]},instructions=transport.WITNESS_INSTRUCTION,choices=transport.WITNESS_CHOICES,scope='Same fixed arm-blind witness questions and code as earlier focused experiment. Valid top3 actionable claims only. Same-model/provider diagnostic, not independent gold, not online timing, no previous grades/verdict supplied.3workers,no retries,no heldout.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=frozen:raise ValueError('Changed concrete witness audit')
    marker.write_text(json.dumps(frozen,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(transport.one,[(r,jobs[r['query_id']]) for r in rows]))
    common={r['query_id'] for r in rows if all(a['review']['status']=='structurally_valid' for a in r['arms'].values())};summary={}
    for scope in ['all','paired_valid']:
        counts={a:Counter() for a in ['focused','concrete']}
        for row in results:
            if scope=='paired_valid' and row['query_id'] not in common:continue
            for name,meta in row['origins'].items():counts[meta['arm']][row.get('audit',{}).get('labels',{}).get(name,'unresolved')]+=1
        summary[scope]={a:dict(c) for a,c in counts.items()}
    report=dict(queries=len(rows),paired_valid_queries=len(common),results=summary,scope=frozen['scope']);(ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
