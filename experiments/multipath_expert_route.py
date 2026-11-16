"""Fixed per-requirement expert recall with matched-size whole-demand control."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_tree_routing import routing_inputs
from expert_search.qwen_embedding import QwenEmbedding


def run():
    trees,items,queries,catalog=routing_inputs();parent=Path('artifacts/expert-tree-routing-qwen-v1')
    spec=json.loads((parent/'protocol.json').read_text())
    if hashlib.sha256(json.dumps([items,queries],sort_keys=True).encode()).hexdigest()!=spec['input_sha256']:raise ValueError('Changed frozen routing matrix order')
    base=np.load(parent/'vectors.npy',allow_pickle=False)
    out=Path('artifacts/multipath-expert-route-v1');out.mkdir(exist_ok=True)
    plans={q['id']:json.loads((Path('artifacts/routing-requirements-v1')/(q['id']+'.json')).read_text()) for q in queries}
    if any(p['status']!='structurally_valid' for p in plans.values()):raise ValueError('Unresolved requirement plan')
    protocol=dict(version=1,per_requirement_experts=2,input_sha256=spec['input_sha256'],plans_sha256=hashlib.sha256(json.dumps(plans,sort_keys=True).encode()).hexdigest(),
        encoder='Qwen3 Embedding 0.6B, frozen existing checkpoint, 4096 token',conditions=['all_nodes','audited_nodes'],
        scope='Fixed top2 experts per demand-only requirement, union before final verification. Compare whole-demand topK with identical K. Two reused silver positive cases; not population recall.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed multipath protocol')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenEmbedding(max_length=4096,batch_size=8);rows=[]
    for index,query in enumerate(queries):
        requirements=plans[query['id']]['requirements'];start=time.perf_counter();vectors=model.encode([r['text'] for r in requirements],query=True);encoding=time.perf_counter()-start
        for condition in protocol['conditions']:
            start=time.perf_counter();positions=[i for i,r in enumerate(items) if r['kind']=='node' and (condition=='all_nodes' or r['verified'])]
            selected=set();paths=[]
            for r,q in zip(requirements,vectors):
                node_scores=base[positions]@q;scores={eid:float('-inf') for eid in trees};best={}
                for i,value in zip(positions,node_scores):
                    eid=items[i]['expert_id']
                    if float(value)>scores[eid]:scores[eid]=float(value);best[eid]=items[i]
                ranked=sorted(trees,key=lambda e:(-scores[e],e));picked=ranked[:2];selected.update(picked)
                paths.append(dict(requirement=r['id'],expert_ids=picked,nodes={eid:best[eid]['id'] for eid in picked},source_ids={eid:best[eid]['source_ids'] for eid in picked}))
            whole_scores=base[positions]@base[len(items)+index];scores={eid:float('-inf') for eid in trees}
            for i,value in zip(positions,whole_scores):scores[items[i]['expert_id']]=max(scores[items[i]['expert_id']],float(value))
            whole=sorted(trees,key=lambda e:(-scores[e],e))[:len(selected)]
            sources={sid for path in paths for ids in path['source_ids'].values() for sid in ids}
            rows.append(dict(query_id=query['id'],condition=condition,requirements=len(requirements),candidate_experts=len(selected),selected_expert_ids=sorted(selected),
                known_expert_retained=query['expert_id'] in selected,whole_demand_same_budget_retained=query['expert_id'] in whole,
                selected_source_count=len(sources),known_source_retained=query['known_source'] in sources,paths=paths,query_encoding_seconds=encoding,cached_route_seconds=time.perf_counter()-start))
    report=dict(protocol=protocol,experts=len(trees),rows=rows,
        warning='Planning latency not recorded in this v1 pilot and must count in future end-to-end tests. No final source verification or applicability ranking included. Additional selected experts may also be relevant; no precision claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    for r in rows:print({k:r[k] for k in ('query_id','condition','candidate_experts','known_expert_retained','whole_demand_same_budget_retained','selected_source_count','known_source_retained')})


if __name__=='__main__':run()
