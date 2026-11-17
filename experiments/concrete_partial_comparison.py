"""Focused bridge instruction ablation on the next20 fixed development queries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from expert_search.focused_partial_match import FocusedPartialMethodMatcher
from expert_search.concrete_partial_match import ConcreteFocusedPartialMatcher,CONCRETE_INSTRUCTION

ROOT=Path('artifacts/concrete-partial-comparison-v1')


def one(item):
    index,job=item;path=ROOT/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    methods=dict(focused=FocusedPartialMethodMatcher(client),concrete=ConcreteFocusedPartialMatcher(client));arms={};order=['focused','concrete'] if index%2==0 else ['concrete','focused']
    for arm in order:
        start=time.perf_counter()
        try:review=methods[arm].match(job['demand'],job['candidates'],k=3)
        except Exception as error:review=dict(status='unresolved',error_type=type(error).__name__)
        arms[arm]=dict(review=review,seconds=time.perf_counter()-start)
    result=dict(query_id=job['id'],split='development',order=order,candidates=list(job['candidates']),arms=arms)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(job['id'],[(a,round(r['seconds'],2),r['review']['status']) for a,r in arms.items()],flush=True);return result


def run():
    source=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs=[j for j in map(json.loads,source.read_text().splitlines()) if j['split']=='development'][40:60]
    ROOT.mkdir(exist_ok=True)
    protocol=dict(model='gpt-6-luna',query_ids=[j['id'] for j in jobs],jobs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection='Fixed development positions40:60, distinct from previous two20query timing cohorts; not sealed or independent human gold.',comparison='Both explain only top3 from same7 raw views; same2400token cap and output schema, one call each, no retries. Concrete arm only adds source-operation/target-operation compatibility and complete Chinese-expression instructions. Alternate arm order,3workers.',added_instruction=CONCRETE_INSTRUCTION,code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/expert_search/concrete_partial_match.py'),Path('src/expert_search/focused_partial_match.py'),Path('src/expert_search/partial_match_schema.py'),Path('src/expert_search/partial_method_match.py')]},scope='Prompt/representation ablation, no claim of novel search algorithm, training or independent accuracy. Same-model silver and witness audits are diagnostic. Heldout unused.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed frozen concrete experiment')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,enumerate(jobs)))
    print('Completed',len(results),'paired queries',flush=True)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
