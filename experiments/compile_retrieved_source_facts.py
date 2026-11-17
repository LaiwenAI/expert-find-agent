"""Warm-cache pilot; source selection is query-derived, compiler content is query-blind."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from expert_search.expert_catalog import ExpertCatalog
from expert_search.source_fact_cards import compile_source_fact_cards,PROMPT

ROOT=Path('artifacts/retrieved-source-facts-v1')


def compile_one(source):
    path=ROOT/'cards'/(hashlib.sha256(source['id'].encode()).hexdigest()+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    result=compile_source_fact_cards(source,LaiwenAI(model='gpt-6-luna',max_retries=0))
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(path)
    print(source['id'],result['status'],len(result['cards']),flush=True);return result


def run():
    parent=Path('artifacts/global-owned-effort-v1/evaluation.json')
    queries_path=Path('artifacts/demand-first-paper-retrieval-v1/queries.jsonl')
    routes=json.loads(parent.read_text());queries={q['id']:q for q in map(json.loads,queries_path.read_text().splitlines())}
    eids={h['id'] for row in routes for h in row['results']['ef512_floor']['hits'][:3]};catalog=ExpertCatalog()
    inventory={s['id']:dict(s,text=s['text'][:1200]) for s in catalog.for_experts(eids)}
    jobs=[];selected={}
    for row in routes:
        candidates={}
        for hit in row['results']['ef512_floor']['hits'][:3]:
            sources={}
            for ref in hit['evidence']:
                source=inventory[ref['id']]
                if source['expert_id']!=hit['id']:raise ValueError('Changed retrieval/source ownership')
                sources[source['id']]=source;selected[source['id']]=source
            candidates[hit['id']]=sources
        jobs.append(dict(id=row['id'],demand=queries[row['id']]['demand'],candidates=candidates))
    (ROOT/'cards').mkdir(parents=True,exist_ok=True)
    protocol=dict(model='gpt-6-luna',prompt=PROMPT,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),queries_sha256=hashlib.sha256(queries_path.read_bytes()).hexdigest(),
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/compile_retrieved_source_facts.py','src/expert_search/source_fact_cards.py']},
                  sources={sid:dict(expert_id=s['expert_id'],sha256=hashlib.sha256(s['text'].encode()).hexdigest()) for sid,s in sorted(selected.items())},
                  scope='All20fixed reused full demands, top3 experts/top3 owned source prefixes from full-corpus ef512 route. Compiler sees one source only, no demand/query/label.4workers,no retries. Nonexhaustive cached pilot, not a complete offline corpus index; cold and whole-corpus costs unmeasured.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed source-fact pilot')
    marker.write_text(json.dumps(protocol,indent=2))
    body=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
    jp=ROOT/'jobs.jsonl'
    if jp.exists() and jp.read_text()!=body:raise ValueError('Changed candidate source jobs')
    jp.write_text(body)
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(compile_one,[selected[k] for k in sorted(selected)]))
    summary=dict(queries=len(jobs),experts=len(eids),sources=len(selected),screened_cards=sum(len(r['cards']) for r in results),
                 sources_with_cards=sum(bool(r['cards']) for r in results),raw_fallback_sources=sum(not r['cards'] for r in results),sum_offline_request_seconds=sum(r['offline_seconds'] for r in results),
                 warning='Four concurrent workers: request-duration sum not wall time. Facts screened by same model/provider; source prefixes and card claims nonexhaustive. Query-derived cohort, no observed full-corpus preprocessing or repeat-query amortization.')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
