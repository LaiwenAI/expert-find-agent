"""Same-budget round-robin multipath routing on the frozen full source index."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.progressive_dense import top_indices

TASK='Find researchers whose documented research background could help address the technical requirement.'


def run():
    parent=Path('artifacts/plan-route-comparison-v1');cases=[json.loads(l) for l in (parent/'cases.jsonl').read_text().splitlines()]
    index=Path('artifacts/global-expert-branch-benchmark-v1');spec=json.loads((index/'protocol.json').read_text());summary=json.loads((index/'summary.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['roots_sha256']:raise ValueError('Changed branch order')
    records=[json.loads(l) for l in source.read_text().splitlines()];ids=sorted({r['expert_id'] for r in records});positions={eid:i for i,eid in enumerate(ids)};owners=np.array([positions[r['expert_id']] for r in records],dtype=np.int32)
    with (index/'vectors.npy').open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=summary['vectors_sha256']:raise ValueError('Changed vector snapshot')
    matrix=np.load(index/'vectors.npy',mmap_mode='r',allow_pickle=False);out=parent/'results';out.mkdir(exist_ok=True)
    protocol=dict(version=1,cases_sha256=hashlib.sha256((parent/'cases.jsonl').read_bytes()).hexdigest(),index_sha256=summary['vectors_sha256'],budgets=[20,50,100,200],
        modes=['whole','requirements','detailed_hypotheses','compact_hypotheses'],fusion='Round-robin over whole-demand path first, then each selected query path, skipping duplicate experts; one source maximum per expert per path.',
        scope='Four incomplete known silver positives among nine frozen cases, new and old groups separate. No precision; inherited planning measurements are historical descriptive runs.')
    marker=parent/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed route comparison protocol')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenTorchEmbedding(max_length=4096,batch_size=8);rows=[]
    for case in cases:
        target=out/(case['id']+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            variants=dict(whole=[case['demand']],requirements=[case['demand']]+[r['text'] for r in case['detailed']],detailed_hypotheses=[case['demand']]+[r['method_hypothesis']['query'] for r in case['detailed']],compact_hypotheses=[case['demand']]+[r['method_hypothesis']['query'] for r in case['compact']]);cache={};results={}
            for mode,queries in variants.items():
                paths=[];encoding=0.;retrieval=0.
                for query in queries:
                    if query not in cache:
                        start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery:'+query],query=False)[0];encoding_seconds=time.perf_counter()-start
                        start=time.perf_counter();scores=matrix@q;maximum=np.full(len(ids),-np.inf,dtype=np.float32);np.maximum.at(maximum,owners,scores)
                        chosen=top_indices(maximum,ids,max(protocol['budgets']));best={}
                        for i in np.flatnonzero(scores==maximum[owners]):best.setdefault(int(owners[i]),int(i))
                        cache[query]=dict(ranking=[ids[i] for i in chosen],source_ids={ids[i]:records[best[i]]['id'] for i in chosen},encoding_seconds=encoding_seconds,retrieval_seconds=time.perf_counter()-start)
                    path=cache[query];paths.append(path);encoding+=path['encoding_seconds'];retrieval+=path['retrieval_seconds']
                selected=[];seen=set()
                for rank in range(max(protocol['budgets'])):
                    for path in paths:
                        eid=path['ranking'][rank]
                        if eid not in seen:selected.append(eid);seen.add(eid)
                        if len(selected)==max(protocol['budgets']):break
                    if len(selected)==max(protocol['budgets']):break
                results[mode]=dict(paths=len(paths),ranking=selected,known_expert_rank=selected.index(case['known_expert_id'])+1 if case['known_expert_id'] in selected else None,
                    retention={str(k):case['known_expert_id'] in selected[:k] for k in protocol['budgets']},encoding_seconds=encoding,retrieval_seconds=retrieval,
                    known_expert_source_ids=sorted({p['source_ids'][case['known_expert_id']] for p in paths if case['known_expert_id'] in p['source_ids']}))
            row=dict(id=case['id'],evaluation_scope=case['evaluation_scope'],results=results,planning_seconds=dict(detailed=case['detailed_planning_seconds'],compact=case['compact_planning_seconds']))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(case['id'],case['evaluation_scope'],{m:r['known_expert_rank'] for m,r in row['results'].items()},flush=True)
    report=dict(cases=len(cases),silver_cases=sum(c['evaluation_scope']!='not_evaluable' for c in cases),
        retention={group:{mode:{str(k):sum(r['results'][mode]['retention'][str(k)] for r in rows if r['evaluation_scope']==group) for k in protocol['budgets']} for mode in protocol['modes']} for group in ('old_development','new_silver')},
        warning='Same-budget candidate routing only. No final source verification, semantic applicability precision or complete relevance gold. Query components reused within comparison; mode timing sums their measured costs, not repeated end-to-end runs.')
    (parent/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
