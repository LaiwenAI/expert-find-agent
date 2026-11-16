"""Typed Decisions alternative to verbose directory navigation, same old cases."""
import hashlib
import inspect
import json
from pathlib import Path
from expert_search.typed_node_router import TypedNodeRouter


def run():
    trees=[json.loads(p.read_text()) for p in sorted(Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json'))];items=[];mapping={}
    for tree in trees:
        audit=json.loads((Path('artifacts/expert-tree-node-audit-v1')/(tree['expert_id']+'.json')).read_text())
        supported={j['id'] for j in audit.get('response',{}).get('judgments',[]) if j['label']=='supported'}
        for node in tree.get('nodes',[]):
            if node['id'] not in supported:continue
            nid='n'+str(len(items));items.append(dict(id=nid,description=json.dumps({k:node[k] for k in ('title','research_problem','methods','objects')},ensure_ascii=False)));mapping[nid]=dict(expert_id=tree['expert_id'],node=node)
    jobs={j['id']:j for j in map(json.loads,Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    keys=sorted({r['parent'] for r in map(json.loads,Path('artifacts/global-expert-roots-v1/queries.jsonl').read_text().splitlines())})
    out=Path('artifacts/typed-expert-tree-route-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,router_sha256=hashlib.sha256(inspect.getsource(TypedNodeRouter).encode()).hexdigest(),input_sha256=hashlib.sha256(json.dumps([items,[jobs[k]['demand'] for k in keys]],sort_keys=True).encode()).hexdigest(),
        scope='70 audited directory nodes, two reused development queries, max node ordinal score per expert. Not final ranking; not a paired latency ablation against 95-node chat router.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed typed routing inputs')
    marker.write_text(json.dumps(protocol,indent=2));router=TypedNodeRouter();rows=[]
    for key in keys:
        target=out/(key+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            row=dict(id=key,status='unresolved')
            try:
                response=router.route(jobs[key]['demand'],items);scores={}
                for j in response['judgments']:
                    eid=mapping[j['id']]['expert_id'];scores[eid]=max(scores.get(eid,float('-inf')),j['routing_score'])
                ranking=sorted(scores,key=lambda e:(-scores[e],e));eid=jobs[key]['expert_id']
                row.update(status='valid',response=response,ranking=ranking,known_expert_rank=ranking.index(eid)+1 if eid in ranking else None)
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(key,row['status'],row.get('known_expert_rank'),flush=True)
    report=dict(queries=len(keys),valid=sum(r['status']=='valid' for r in rows),nodes=len(items),
        request_seconds=sum(r.get('response',{}).get('seconds',0) for r in rows),top5_known_retained=sum(r.get('known_expert_rank') is not None and r['known_expert_rank']<=5 for r in rows),
        warning='Two reused incomplete silver cases only. Model applicability routes need original source verification; no final expert-fit precision or project-success claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
