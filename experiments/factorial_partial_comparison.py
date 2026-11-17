"""2x2 language/bridge instruction ablation on fixed final20 dev queries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from expert_search.focused_partial_match import FocusedPartialMethodMatcher
from expert_search.concrete_partial_match import CONCRETE_INSTRUCTION

ROOT=Path('artifacts/factorial-partial-comparison-v1')
LANGUAGE,marker,tail=CONCRETE_INSTRUCTION.partition(' For each d/t')
if not marker:raise ValueError('Cannot separate instruction factors')
BRIDGE='For each d/t'+tail
INSTRUCTIONS=dict(base='',language=LANGUAGE,bridge=BRIDGE,both=CONCRETE_INSTRUCTION)
ARMS=list(INSTRUCTIONS)
ORDERS=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]


def matcher(client,instruction):
    if not instruction:return FocusedPartialMethodMatcher(client)
    def create(**kwargs):
        messages=[dict(m) for m in kwargs['messages']];messages[0]['content']+='\n'+instruction
        return client.chat.completions.create(**dict(kwargs,messages=messages))
    return FocusedPartialMethodMatcher(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))


def one(item):
    index,job=item;path=ROOT/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    client=LaiwenAI(model='gpt-6-luna',max_retries=0);results={};order=[ARMS[i] for i in ORDERS[index%4]]
    for arm in order:
        start=time.perf_counter()
        try:review=matcher(client,INSTRUCTIONS[arm]).match(job['demand'],job['candidates'],k=3)
        except Exception as error:review=dict(status='unresolved',error_type=type(error).__name__)
        results[arm]=dict(review=review,seconds=time.perf_counter()-start)
    result=dict(query_id=job['id'],split='development',order=order,candidates=list(job['candidates']),arms=results)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(job['id'],[(a,round(r['seconds'],2),r['review']['status']) for a,r in results.items()],flush=True);return result


def run():
    source=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs=[j for j in map(json.loads,source.read_text().splitlines()) if j['split']=='development'][60:80]
    if len(jobs)!=20:raise ValueError('Wrong final development cohort')
    ROOT.mkdir(exist_ok=True)
    protocol=dict(model='gpt-6-luna',query_ids=[j['id'] for j in jobs],jobs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection='Fixed final20development positions60:80; distinct from previous60query engineering cohorts. Not sealed/independent human gold.',factors=INSTRUCTIONS,order='Four-arm Williams Latin-square order, each of4orders used5times; balanced positions and immediate carryover.3workers,no retries. Same7raw inputs,top3outputs,2400cap/schema,reasoningnone.',code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/expert_search/concrete_partial_match.py'),Path('src/expert_search/focused_partial_match.py'),Path('src/expert_search/partial_match_schema.py'),Path('src/expert_search/partial_method_match.py')]},scope='Prompt/representation factorial diagnostic, not a novel search algorithm, training or independent expert-accuracy benchmark. Existing heldout20 excluded.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed frozen factorial protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,enumerate(jobs)))
    print('Completed',len(results),'four-arm queries',flush=True)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
