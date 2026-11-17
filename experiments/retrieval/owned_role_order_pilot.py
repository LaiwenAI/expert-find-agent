"""Paired forward/reverse native choices on the same frozen source views."""
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.owned_role_decision_order import decide_roles
from .owned_role_decision_pilot import sha


def main():
    prior=Path('artifacts/owned-role-paper-pilot-v1');p=json.loads((prior/'protocol.json').read_text());data=prior/'sources.json'
    if sha(data)!=p['source_sha256']:raise ValueError('Changed source views')
    sources=json.loads(data.read_text())
    if len(sources)!=p['sources']:raise ValueError('Changed source count')
    rng=random.Random(20261125);schedule=[]
    for i in range(len(sources)):
        orders=['forward','reverse'];rng.shuffle(orders)
        schedule.extend({'index':i,'order':order} for order in orders)
    paths=['experiments/retrieval/owned_role_order_pilot.py','src/expert_search/owned_role_decision_order.py',
           'src/expert_search/owned_role_decision.py','src/expert_search/owned_role_windows.py',
           'src/expert_search/source_roles.py','src/expert_search/choice_judgments.py']
    root=Path('artifacts/owned-role-order-pilot-v1');root.mkdir(exist_ok=False)
    protocol={'parent_protocol_sha256':sha(prior/'protocol.json'),'source_sha256':sha(data),
              'code_sha256':{name:sha(Path(name)) for name in paths},'schedule':schedule,'seed':20261125,
              'width':p['width'],'sources':len(sources),'model':'gpt-6-luna',
              'scope':'Post-hoc, paired forward/reverse choice order on same 12 paper views; random order within each pair. One attempt per order. Agreement is not correctness; no independent human labels or broad latency claim.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    load_dotenv('.env');client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    for item in schedule:
        i,order=item['index'],item['order'];result=decide_roles(sources[i],client,order,width=p['width'])
        result['source_index']=i
        (root/(f'source-{i+1:03d}-{order}.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f'source-{i+1:03d}',order,result['status'],flush=True)


if __name__=='__main__':main()
