"""Matched coarse-routing diagnostic on two known silver demand/expert pairs."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.expert_catalog import ExpertCatalog


def routing_inputs():
    trees={r['expert_id']:r for p in Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json') if (r:=json.loads(p.read_text()))['status']=='structurally_valid'}
    audited={r['expert_id']:{j['id']:j['label'] for j in r['response']['judgments']} for p in Path('artifacts/expert-tree-node-audit-v1').glob('CN_*.json') if (r:=json.loads(p.read_text()))['status']=='valid'}
    catalog=ExpertCatalog();items=[]
    for eid,tree in sorted(trees.items()):
        for source in catalog.for_experts([eid]):
            if source.get('field') in ('研究领域_CN','研究领域_EN'):
                items.append(dict(id=source['id'],expert_id=eid,kind='area',text=source['text'],source_ids=[source['id']]))
        for node in tree['nodes']:
            items.append(dict(id=node['id'],expert_id=eid,kind='node',verified=audited[eid][node['id']]=='supported',
                text=json.dumps({k:node[k] for k in ('title','research_problem','methods','objects')},ensure_ascii=False),
                source_ids=sorted({c['evidence_id'] for c in node['citations']})))
    jobs={j['id']:j for j in map(json.loads,Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    summary=json.loads(Path('artifacts/paper-control-blind-audit-v1/summary.json').read_text());queries=[]
    for row in summary['rows']:
        if row['status']!='dual_model_control_candidate':continue
        job=jobs[row['id']];audit=json.loads((Path('artifacts/paper-control-blind-audit-v1')/(row['id']+'-full-gpt-4o.json')).read_text())['audit']
        queries.append(dict(id=row['id'],text=job['demand'],expert_id=job['expert_id'],known_source=audit['citation']['evidence_id']))
    return trees,items,queries,catalog


def run(qwen=False):
    trees,items,queries,catalog=routing_inputs()
    out=Path('artifacts/expert-tree-routing-qwen-v1' if qwen else 'artifacts/expert-tree-routing-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,model='Qwen/Qwen3-Embedding-0.6B' if qwen else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',max_length=4096 if qwen else 128,
        input_sha256=hashlib.sha256(json.dumps([items,queries],sort_keys=True).encode()).hexdigest(),budgets=[1,3,5,10],
        aggregation='Maximum cosine across routing entries owned by an expert; missing entries negative infinity. Original full enterprise demands as queries.',
        conditions=['area','all_nodes','audited_nodes'],scope='Two known silver positive pairs, same 21-expert cohort. Not complete qrels, precision or unseen-query evaluation.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed routing protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    if qwen:
        from expert_search.qwen_embedding import QwenEmbedding
        model=QwenEmbedding(max_length=4096,batch_size=8)
    else:
        from fastembed import TextEmbedding
        model=TextEmbedding(model_name=protocol['model'],cache_dir='artifacts/models',threads=1,max_length=128)
    vp=out/'vectors.npy'
    if vp.exists():vectors=np.load(vp,allow_pickle=False)
    else:
        if qwen:
            vectors=np.concatenate([model.encode([r['text'] for r in items]),model.encode([q['text'] for q in queries],query=True)])
        else:vectors=np.asarray(list(model.embed([r['text'] for r in items]+[q['text'] for q in queries])),dtype=np.float32)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True);np.save(vp,vectors)
    if vectors.shape!=(len(items)+len(queries),1024 if qwen else 384) or not np.isfinite(vectors).all():raise ValueError('Invalid routing vectors')
    rows=[]
    for i,query in enumerate(queries):
        scores=vectors[:len(items)]@vectors[len(items)+i]
        for condition in protocol['conditions']:
            begin=time.perf_counter();positions=[k for k,r in enumerate(items) if (r['kind']=='area' if condition=='area' else r['kind']=='node' and (condition=='all_nodes' or r['verified']))]
            expert_scores={eid:float('-inf') for eid in trees};evidence=[]
            for k in positions:expert_scores[items[k]['expert_id']]=max(expert_scores[items[k]['expert_id']],float(scores[k]))
            ranking=sorted(trees,key=lambda eid:(-expert_scores[eid],eid));node_order=sorted(positions,key=lambda k:(-float(scores[k]),items[k]['id']))
            rows.append(dict(query_id=query['id'],condition=condition,known_expert_rank=ranking.index(query['expert_id'])+1,
                first_known_source_node_rank=next((rank for rank,k in enumerate(node_order,1) if items[k]['expert_id']==query['expert_id'] and query['known_source'] in items[k]['source_ids']),None),
                retention={str(k):query['expert_id'] in ranking[:k] for k in protocol['budgets']},cached_routing_seconds=time.perf_counter()-begin))
    report=dict(protocol=protocol,experts=len(trees),routing_entries=len(items),known_pairs=len(queries),rows=rows,
        retention={condition:{str(k):sum(r['retention'][str(k)] for r in rows if r['condition']==condition) for k in protocol['budgets']} for condition in protocol['conditions']},
        vectors_sha256=hashlib.sha256(vp.read_bytes()).hexdigest(),warning='Cached query routing only; no online encoding latency. Not precision or true global recall. Dense similarity and model-audited summaries are not applicability judgments.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--qwen',action='store_true');run(parser.parse_args().qwen)
