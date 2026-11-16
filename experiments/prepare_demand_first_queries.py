"""Freeze the original fixed-random demand cohort without outcome filtering."""
import hashlib
import json
from pathlib import Path


def run():
    source=Path('artifacts/demands/queries.jsonl');rows=[json.loads(l) for l in source.read_text().splitlines()]
    if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('Changed fixed twenty-demand cohort')
    out=Path('artifacts/demand-first-paper-retrieval-v1');out.mkdir(exist_ok=True)
    body=''.join(json.dumps(dict(id=r['id'],demand=r['query']),ensure_ascii=False)+'\n' for r in rows)
    selection=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),queries=len(rows),selection='All original fixed-random 20 demands, unchanged original order. First paper-max expert and its top three source prefixes. No teacher-topic overlap filter or outcome selection.',warning='Reused exploratory cohort, not held-out. No relevance labels.')
    for name,text in [('queries.jsonl',body),('selection.json',json.dumps(selection,indent=2))]:
        target=out/name
        if target.exists() and target.read_text()!=text:raise ValueError('Changed frozen query selection')
        target.write_text(text)
    print('Prepared',len(rows),'fixed demands')


if __name__=='__main__':run()
