"""Export raw-derived, unjudged development pool additions for future review."""
import hashlib
import json
from pathlib import Path


def run():
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');audit=Path('artifacts/fresh-lexical-pool-audit-v1')
    runs_path=audit/'runs.jsonl';summary=json.loads((audit/'summary.json').read_text())
    if hashlib.sha256(runs_path.read_bytes()).hexdigest()!=summary['runs_sha256']:raise ValueError('Changed discovery runs')
    queries={q['id']:q for q in map(json.loads,(root/'queries.jsonl').read_text().splitlines()) if q['split']=='development'}
    additions={}
    for row in map(json.loads,runs_path.read_text().splitlines()):
        if row['query_id'] not in queries:raise ValueError('Non-development query')
        for ranked in row['ranking']:
            if ranked['expert_id'] not in row['outside_frozen_pool']:continue
            entry=additions.setdefault((row['query_id'],ranked['expert_id']),dict(source_ids=[],origins=[]))
            entry['origins'].append(row['variant'])
            for sid in ranked['source_ids']:
                if sid not in entry['source_ids']:entry['source_ids'].append(sid)
    needed={s for a in additions.values() for s in a['source_ids'][:3]};sources={}
    for source in map(json.loads,(root/'corpus.jsonl').read_text().splitlines()):
        if source['id'] in needed:sources[source['id']]=source
    if set(sources)!=needed:raise ValueError('Missing raw-derived source')
    experts={e['id']:e for e in map(json.loads,(root/'experts.jsonl').read_text().splitlines())}
    rows=[];origins=[]
    for (qid,eid),entry in sorted(additions.items()):
        views=[]
        for sid in entry['source_ids'][:3]:
            source=sources[sid]
            if source['expert_id']!=eid or source['split']!='development':raise ValueError('Foreign source or split')
            text=source['text'][:1200];views.append(dict(source_id=sid,text=text,start=0,end=len(text),sha256=hashlib.sha256(text.encode()).hexdigest(),provenance=source['provenance']))
        snapshot_id=hashlib.sha256(json.dumps([qid,eid,views],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        rows.append(dict(query_id=qid,expert_id=eid,snapshot_id=snapshot_id,split='development',demand=queries[qid]['text'],expert=experts[eid],sources=views,grade=None,human_grade=None,expert_ability=None,future_project_success=None,label_status='unjudged_new_lexical_view'))
        origins.append(dict(snapshot_id=snapshot_id,origins=entry['origins']))
    out=audit/'review-additions';out.mkdir(exist_ok=True)
    for name,data in [('human-review.jsonl',rows),('retrieval-origins.jsonl',origins)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    report=dict(queries=len({r['query_id'] for r in rows}),candidate_snapshots=len(rows),experts=len({r['expert_id'] for r in rows}),sources=len(needed),labels=0,heldout_queries_used=0,
                scope='Local unjudged development additions. First3 unique BM25 witnesses, full-demand then title order, each first1200chars. Review file omits retrieval origins; author identity remains visible. Views differ from frozen benchmark, so no inherited labels. Not a claim of relevance or semantic correctness.',
                parent_dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),runs_sha256=summary['runs_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.jsonl'))})
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
