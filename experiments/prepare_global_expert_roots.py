"""Source-linked coarse expert roots for full-corpus routing evaluation."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from expert_search.corpus import records


def run():
    corpus=Path('artifacts/corpus');experts={e['id']:e for _,e in records(corpus/'experts.jsonl')};areas=defaultdict(list);titles=defaultdict(list);fallback=defaultdict(list)
    for _,e in records(corpus/'evidence.jsonl'):
        if e['text'].strip() and len(fallback[e['expert_id']])<1:fallback[e['expert_id']].append(e)
        if e.get('field') in ('\u7814\u7a76\u9886\u57df_CN','\u7814\u7a76\u9886\u57df_EN') and e['text'].strip():areas[e['expert_id']].append(e)
        if e['kind']=='paper' and e.get('title') and len(titles[e['expert_id']])<3:titles[e['expert_id']].append(e)
    rows=[];unavailable=[]
    for eid in sorted(experts):
        selected=areas[eid]+titles[eid]
        if not selected:selected=fallback[eid]
        if not selected:unavailable.append(eid);continue
        views=[dict(evidence_id=e['id'],text=e['title'] if e in titles[eid] else e['text'][:1000],role='area' if e in areas[eid] else 'paper_title' if e in titles[eid] else 'raw_source_fallback') for e in selected]
        rows.append(dict(expert_id=eid,text='\n'.join(v['text'] for v in views),source_views=views))
    out=Path('artifacts/global-expert-roots-v1');out.mkdir(exist_ok=True);body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'roots.jsonl').write_text(body)
    with (corpus/'evidence.jsonl').open('rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(version=1,total_experts=len(experts),indexed_experts=len(rows),unavailable_experts=unavailable,
        roots_sha256=hashlib.sha256(body.encode()).hexdigest(),source_sha256=source_sha,
        policy='All available CN/EN research-area prefixes (1000 chars each) plus first three available paper titles in frozen source order. If neither exists, retain first nonempty raw profile/abstract prefix as fallback. No generated capability facts, no demand or relevance labels used.',
        planned_candidate_budgets=[20,50,100,200,500],warning='Missing root metadata forms an explicit unknown bucket, not negative expertise. Roots are coarse routing hints, never ability proof.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='unavailable_experts'},indent=2))


if __name__=='__main__':run()
