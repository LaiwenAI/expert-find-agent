"""Fair-format raw/card candidate comparison; model-stage replay, not end-to-end."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from expert_search.partial_method_match import PartialMethodMatcher,PROMPT


def run():
    parent=Path('artifacts/retrieved-source-facts-v1');jobs_path=parent/'jobs.jsonl'
    jobs=[json.loads(s) for s in jobs_path.read_text().splitlines()]
    card_files=sorted((parent/'cards').glob('*.json'));cards={}
    for path in card_files:
        result=json.loads(path.read_text());cards[result['source_id']]=result
    if len(cards)!=len(json.loads((parent/'protocol.json').read_text())['sources']):raise ValueError('Incomplete source-card cohort')
    root=Path('artifacts/partial-method-comparison-v1');root.mkdir(exist_ok=True)
    protocol=dict(model='gpt-6-luna',prompt=PROMPT,source_jobs_sha256=hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                  compiler_protocol_sha256=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest(),card_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in card_files},
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/compare_partial_method_match.py','src/expert_search/partial_method_match.py']},
                  modes=['raw','cards'],max_calls_per_mode=2,selection='All20 fixed full demands, same top3 full-corpus candidates/top3source prefixes, candidate order shuffled identically per query; mode order alternates. Sequential API requests, no retries; all failures/partial reads retained.',
                  scope='Comparison of partial method-fit model outputs, same format; no full requirement coverage score. Replayed retrieval/cached selected-source cards, so not fresh end-to-end or whole-corpus preprocessing. Offline cost and nonexhaustiveness separate. Same model/provider, no independent expert-fit gold; agreement is not accuracy.')
    marker=root/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed partial matcher comparison')
    marker.write_text(json.dumps(protocol,indent=2))
    from laiwenai import LaiwenAI
    matcher=PartialMethodMatcher(LaiwenAI(model='gpt-6-luna',max_retries=0));rows=[]
    for i,job in enumerate(jobs):
        pairs=list(job['candidates'].items());random.Random(job['id']).shuffle(pairs);candidates=dict(pairs)
        order=['raw','cards'] if i%2==0 else ['cards','raw'];results={}
        for mode in order:
            path=root/(job['id']+'-'+mode+'.json')
            if path.exists():result=json.loads(path.read_text())
            else:
                start=time.perf_counter()
                try:result=matcher.match(job['demand'],candidates,cards,mode)
                except Exception as error:result=dict(status='unresolved',error_type=type(error).__name__,wall_seconds=time.perf_counter()-start,api_attempts=0,rounds=[])
                temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(path)
            results[mode]=result;print(job['id'],mode,result['status'],round(result['wall_seconds'],3),flush=True)
        rows.append(dict(id=job['id'],results=results))
    modes={}
    for mode in ['raw','cards']:
        rs=[r['results'][mode] for r in rows]
        modes[mode]=dict(queries=len(rs),valid=sum(r['status']=='structurally_valid' for r in rs),partial=sum(r['status']=='partial_more_source_needed' for r in rs),unresolved=sum(r['status']=='unresolved' for r in rs),
                         total_api_attempts=sum(r['api_attempts'] for r in rs),mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rs])),
                         mean_total_input_chars=float(np.mean([sum(p['input_chars'] for p in r['rounds']) for r in rs])),
                         total_reported_completion_tokens=sum(p['response'].get('usage',{}).get('completion_tokens',0) for r in rs for p in r['rounds']))
    valid=[r for r in rows if all(r['results'][m]['status']=='structurally_valid' for m in ['raw','cards'])]
    top1=sum(r['results']['raw']['result']['ranking'][0]['expert_id']==r['results']['cards']['result']['ranking'][0]['expert_id'] for r in valid)
    agreements=0;assessments=0
    for row in valid:
        raw={a['expert_id']:a['level'] for a in row['results']['raw']['result']['ranking']}
        for a in row['results']['cards']['result']['ranking']:
            assessments+=1;agreements+=a['level']==raw[a['expert_id']]
    summary=dict(queries=len(rows),methods=modes,both_valid_queries=len(valid),top1_agreement=top1,candidate_level_agreement=agreements,paired_candidate_assessments=assessments,
                 offline_compilation=json.loads((parent/'summary.json').read_text()),
                 warning='Agreement/provisional levels are not expert accuracy. Format/ownership validation not semantic correctness. One sequential run/mode, no repeated stability or independent gold. Failures included in timing. Replayed retrieval, source cache creation, card load and encoder/index initialization excluded; no end-to-end speedup or observed amortization.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
