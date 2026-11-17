"""Compare one- and two-call routing on identical frozen candidate evidence."""
import time


def preserve_candidate_coverage(hits, selected, budget=30):
    """Reserve one evidence item per eligible candidate, then add routed items."""
    eligible=[h for h in hits if h['evidence']]
    if len(eligible)>budget:
        raise ValueError('Budget cannot cover all candidates with evidence')
    chosen=[]
    routed=set(selected)
    for hit in eligible:
        first=next((e['id'] for e in hit['evidence'] if e['id'] in routed),hit['evidence'][0]['id'])
        if first not in chosen:chosen.append(first)
    allowed={e['id'] for hit in hits for e in hit['evidence']}
    for item in selected:
        if item in allowed and item not in chosen and len(chosen)<budget:chosen.append(item)
    return chosen


def run_case(agent, query, hits, arm, k=10, frozen_route=None):
    start=time.perf_counter();agent.trace=[];agent.usage=dict.fromkeys(agent.usage,0)
    evidence={e['id']:e for hit in hits for e in hit['evidence']}
    if arm in ('routed','routed_coverage'):
        if frozen_route is None:
            route=agent.ask('route','Select at most 30 evidence IDs relevant to the demand. Return {"leaf_ids":[str]}; only choose supplied IDs.',{'demand':query,'evidence':[{'id':e['id'],'text':e['text'][:6000]} for e in evidence.values()]})
            route_ids=route.get('leaf_ids',[])
        else:
            route_ids=frozen_route
        selected=list(dict.fromkeys(x for x in route_ids if isinstance(x,str) and x in evidence))[:30]
        if arm=='routed_coverage':selected=preserve_candidate_coverage(hits,selected)
    elif arm=='direct':selected=list(evidence)
    elif arm=='coverage_direct':selected=preserve_candidate_coverage(hits,[])
    else:raise ValueError('Unknown arm')
    if not selected:raise ValueError('No valid selected evidence')
    answer=agent.ask('verify','Rank experts using only supplied evidence. Cite evidence IDs and state missing requirements. Do not infer capability from names or institutions. Return {"results":[{"expert_id":str,"reason":str,"missing":[str],"citations":[str]}]}.',{'demand':query,'evidence':[dict(evidence[x],text=evidence[x]['text'][:6000]) for x in selected],'top_k':k})
    allowed={h['id'] for h in hits};rank=[];seen=set()
    for row in answer.get('results',[]):
        eid=row.get('expert_id');cites=[x for x in row.get('citations',[]) if isinstance(x,str) and x in selected and evidence[x]['expert_id']==eid]
        if eid in allowed and eid not in seen and cites:seen.add(eid);rank.append(eid)
        if len(rank)==k:break
    return dict(arm=arm,ranking=rank,selected_count=len(selected),seconds=time.perf_counter()-start,trace=agent.trace,usage=agent.usage)
