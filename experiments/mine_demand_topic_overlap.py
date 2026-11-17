"""Mine real demand/teacher-topic anchors; literal overlap is not suitability."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from openpyxl import load_workbook
from expert_search.corpus import records


def run():
    source=next(Path('data/\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42').glob('*.xlsx'))
    book=load_workbook(source,read_only=True,data_only=True);sheet=book.active
    iterator=iter(sheet.values);header=next(iterator)
    topics=defaultdict(list)
    for _,e in records('artifacts/corpus/evidence.jsonl'):
        if e.get('field')!='\u7814\u7a76\u9886\u57df_CN':continue
        for phrase in e['text'].splitlines():
            phrase=phrase.strip()
            if 4<=len(phrase)<=24 and all(c.isalnum() or '\u4e00'<=c<='\u9fff' for c in phrase):
                topics[phrase].append(dict(expert_id=e['expert_id'],evidence_id=e['id']))
    first=defaultdict(set)
    for phrase in topics:first[phrase[:3]].add(phrase)
    pairs=[];demands=0
    for line,values in enumerate(iterator,2):
        d=dict(zip(header,values))
        if not d.get('\u547d\u9898\u540d\u79f0') or not d.get('\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42'):continue
        demands+=1;text=str(d['\u547d\u9898\u540d\u79f0'])+'\n'+str(d['\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42'])
        candidates=set()
        for i in range(len(text)-2):candidates.update(first.get(text[i:i+3],()))
        hits=[p for p in candidates if text.count(p)==1]
        # Group before selecting so two overlapping strings cannot supply
        # artificial evidence for two independently covered requirements.
        by_expert=defaultdict(list)
        for phrase in hits:
            for owner in topics[phrase]:by_expert[owner['expert_id']].append(dict(topic=phrase,start=text.index(phrase),end=text.index(phrase)+len(phrase),evidence_id=owner['evidence_id']))
        for eid,matches in by_expert.items():
            unique={m['topic']:m for m in matches};chosen=[]
            for m in sorted(unique.values(),key=lambda m:(-len(m['topic']),m['topic'])):
                if all(m['end']<=x['start'] or m['start']>=x['end'] for x in chosen):chosen.append(m)
            if len(chosen)>=2 or any(len(m['topic'])>=8 for m in chosen):
                pairs.append(dict(demand_id=f'demand-{line}',title=str(d['\u547d\u9898\u540d\u79f0']),demand=text,expert_id=eid,matches=chosen))
    pairs.sort(key=lambda r:(-sum(len(m['topic']) for m in r['matches']),r['demand_id'],r['expert_id']))
    out=Path('artifacts/demand-topic-overlap-v1');out.mkdir(exist_ok=True)
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
    report=dict(demands=demands,distinct_topics=len(topics),candidate_pairs=len(pairs),matched_demands=len({r['demand_id'] for r in pairs}),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),warning='Selection by literal topic overlap only. Company background and future requirements may contain these phrases. Not relevance, capacity, or full-demand gold. Original text kept local.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
