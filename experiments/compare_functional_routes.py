"""Same-node routing view comparison, without capability inference from hints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from expert_search.qwen_embedding import QwenEmbedding


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition-diagnostic",action="store_true",help="Use known silver narrow conditions; not deployment queries")
    args=parser.parse_args()
    source=Path('artifacts/method-routing-views-v1/views.jsonl');views=[json.loads(l) for l in source.read_text().splitlines()]
    nodes={n['id']:n for p in Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json') for n in json.loads(p.read_text()).get('nodes',[])}
    jobs={j['id']:j for j in map(json.loads,Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    queries=[]
    for r in json.loads(Path('artifacts/paper-control-blind-audit-v1/summary.json').read_text())['rows']:
        if r['status']=='dual_model_control_candidate':
            a=json.loads((Path('artifacts/paper-control-blind-audit-v1')/(r['id']+'-full-gpt-4o.json')).read_text())['audit']
            queries.append(dict(id=r['id'],text=jobs[r['id']]['demand'],expert_id=jobs[r['id']]['expert_id'],known_source=a['citation']['evidence_id']))
    if args.condition_diagnostic:
        examples={r["pair_id"]:r for r in map(json.loads,Path("artifacts/dataset-release/paper-controls-combined-v2/examples.jsonl").read_text().splitlines()) if r["view"]=="full"}
        for q in queries:q["text"]=examples[q["id"]]["condition"]
    items=[]
    for v in views:
        n=nodes[v['id']]
        descriptions=dict(original=json.dumps({k:n[k] for k in ('title','research_problem','methods','objects')},ensure_ascii=False),
            signature=json.dumps(v['observed'],ensure_ascii=False),abstract=v['search_hint']['text'])
        for mode,text in descriptions.items():items.append(dict(id=v['id'],expert_id=v['expert_id'],mode=mode,text=text,source_ids=sorted({c['evidence_id'] for c in v['citations']})))
    out=Path('artifacts/functional-route-condition-diagnostic-v1' if args.condition_diagnostic else 'artifacts/functional-route-comparison-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,input_sha256=hashlib.sha256(json.dumps([items,queries],sort_keys=True).encode()).hexdigest(),
        encoder='Official frozen Qwen3 Embedding 0.6B MLX',max_length=4096,batch_size=8,conditions=['original','signature','abstract','original_plus_abstract'],budgets=[3,5,10],
        query_role='known_silver_condition_diagnostic' if args.condition_diagnostic else 'original_enterprise_demand',
        scope='Same 66 nodes and 21 experts; two reused silver development cases. No precision or semantic applicability gold.')
    if not args.condition_diagnostic:protocol.pop('query_role')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed functional routing comparison')
    marker.write_text(json.dumps(protocol,indent=2));vp=out/'vectors.npy'
    if vp.exists():vectors=np.load(vp,allow_pickle=False)
    else:
        model=QwenEmbedding(max_length=4096,batch_size=8)
        vectors=np.concatenate([model.encode([r['text'] for r in items]),model.encode([q['text'] for q in queries],query=True)]);np.save(vp,vectors)
    if vectors.shape!=(len(items)+len(queries),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid functional vectors')
    experts=sorted({v['expert_id'] for v in views});rows=[]
    for i,query in enumerate(queries):
        scores=vectors[:len(items)]@vectors[len(items)+i]
        for condition in protocol['conditions']:
            selected=[k for k,r in enumerate(items) if r['mode']==condition or (condition=='original_plus_abstract' and r['mode'] in ('original','abstract'))]
            expert_scores={eid:float('-inf') for eid in experts}
            for k in selected:expert_scores[items[k]['expert_id']]=max(expert_scores[items[k]['expert_id']],float(scores[k]))
            ranking=sorted(experts,key=lambda eid:(-expert_scores[eid],eid));ranked_nodes=sorted(selected,key=lambda k:(-float(scores[k]),items[k]['id']))
            rows.append(dict(query_id=query['id'],condition=condition,known_expert_rank=ranking.index(query['expert_id'])+1,
                first_known_source_node_rank=next((rank for rank,k in enumerate(ranked_nodes,1) if items[k]['expert_id']==query['expert_id'] and query['known_source'] in items[k]['source_ids']),None),
                retention={str(k):query['expert_id'] in ranking[:k] for k in protocol['budgets']}))
    report=dict(protocol=protocol,experts=len(experts),nodes=len(views),rows=rows,
        warning='Abstract functions remain routing hints only. Query vectors cached; no online latency measured. Four representation views are exploratory, not a tuned confirmed method.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
