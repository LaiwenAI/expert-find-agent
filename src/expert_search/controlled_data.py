"""Exact listed-research-area conjunction diagnostics, separate from semantic expert gold."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from .corpus import records


def build(limit=500,seed=42,output='artifacts/controlled-constraints'):
    areas=defaultdict(set);spans={};sources={}
    for _,r in records('artifacts/corpus/evidence.jsonl'):
        if r.get('field')!='研究领域_CN':continue
        cursor=0
        for line in r['text'].splitlines(keepends=True):
            phrase=line.strip()
            if 3<=len(phrase)<=40 and not any(x in phrase for x in ['http','。','：']):
                areas[phrase].add(r['expert_id'])
                spans[(r['expert_id'],phrase)]=dict(evidence_id=r['id'],start=cursor+line.index(phrase),end=cursor+line.index(phrase)+len(phrase),source=r['source'])
                sources[r['id']]=r['text']
            cursor+=len(line)
    by_expert=defaultdict(list)
    for phrase,experts in areas.items():
        for eid in experts:by_expert[eid].append(phrase)
    candidates=[eid for eid in sorted(by_expert) if len(by_expert[eid])>=2];random.Random(seed).shuffle(candidates)
    rows=[];used=set()
    for eid in candidates:
        phrases=sorted(by_expert[eid]);rng=random.Random(seed+int(hashlib.sha256(eid.encode()).hexdigest()[:8],16));a,b=rng.sample(phrases,2)
        key=tuple(sorted([a,b]))
        if key in used:continue
        used.add(key);support=areas[a]&areas[b];partial=areas[a]^areas[b]
        references=[dict(constraint=phrase,**spans[(eid,phrase)]) for phrase in [a,b]]
        for ref in references:assert sources[ref['evidence_id']][ref['start']:ref['end']]==ref['constraint']
        rows.append(dict(id='conjunction-'+str(len(rows)),query=f'寻找公开研究方向同时明确列有「{a}」与「{b}」的专家',
            constraints=[a,b],known_both=sorted(support),known_one=sorted(partial),construction_expert=eid,references=references,
            label_semantics='Exact phrase membership in indexed research-area lines, not actual ability or semantic relevance'))
        if len(rows)==limit:break
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'queries.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(queries=len(rows),seed=seed,distinct_area_phrases=len(areas),experts_with_areas=len(by_expert),
        all_construction_spans_verified=True,warning='Controlled lexical conjunction diagnostic. Not industrial-query gold; no inferred negative ability labels; do not merge with semantic benchmark scores.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':print(json.dumps(build(),indent=2))
