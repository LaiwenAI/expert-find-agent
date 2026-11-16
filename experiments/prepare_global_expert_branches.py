"""One source-linked routing branch per nonempty attributed evidence record."""
import hashlib
import json
from pathlib import Path
from expert_search.corpus import records


def run():
    source=Path('artifacts/corpus/evidence.jsonl');out=Path('artifacts/global-expert-branches-v1');out.mkdir(exist_ok=True)
    rows=[]
    for _,e in records(source):
        if not e['text'].strip():continue
        text=e['text'][:1200]
        rows.append(dict(id=e['id'],expert_id=e['expert_id'],text=text,kind=e['kind'],
            source_prefix_sha256=hashlib.sha256(text.encode()).hexdigest(),source=e['source']))
    body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(out/'branches.jsonl').write_text(body)
    with source.open('rb') as stream:source_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(version=1,branches=len(rows),experts=len({r['expert_id'] for r in rows}),source_sha256=source_sha,
        branches_sha256=hashlib.sha256(body.encode()).hexdigest(),budgets=[20,50,100,200,500],
        policy='Every nonempty raw evidence record, prefix 1200 chars, one owner; no label or demand based selection. Expert score is maximum branch cosine.',
        warning='Source prefixes are routing hints, not full-document coverage or capability labels. Token truncation remains possible; missing evidence is unknown.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
