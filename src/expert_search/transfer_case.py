"""Resolve conditional transfer proposals without inventing achieved capability."""
def resolve_transfer(job, proposal, demand_spans, source_spans):
    if type(proposal.get('feasible')) is not bool:
        raise ValueError('Missing transfer feasibility')
    if not proposal['feasible']:
        if not isinstance(proposal.get('reason'),str) or not proposal['reason'].strip():
            raise ValueError('Missing rejection reason')
        return dict(status='no_proposal',reason=proposal['reason'])
    bounds={'capability':(10,240),'target_operation':(10,240),'mechanism':(20,480)}
    for key,(low,high) in bounds.items():
        if not isinstance(proposal.get(key),str) or not low<=len(proposal[key])<=high:
            raise ValueError('Invalid transfer field: '+key)
    did=proposal.get('demand_span_id')
    if did not in demand_spans:raise ValueError('Unknown demand span')
    target=demand_spans[did]
    if job['demand'][target['start']:target['end']]!=target['text']:
        raise ValueError('Changed demand provenance')
    for name,maximum in [('assumptions',4),('remaining_demand_gaps',6)]:
        values=proposal.get(name)
        if not isinstance(values,list) or not 1<=len(values)<=maximum or any(not isinstance(v,str) or not v.strip() or len(v)>240 for v in values):
            raise ValueError('Missing conditional assumptions or scope gaps')
    refs=proposal.get('evidence_spans')
    if not isinstance(refs,list) or not 1<=len(refs)<=3:raise ValueError('Invalid transfer witnesses')
    citations=[];seen=set();owners=set()
    for ref in refs:
        sid=ref.get('evidence_id');span_id=ref.get('evidence_span_id');source=job['sources'].get(sid)
        span=source_spans.get(sid,{}).get(span_id)
        if source is None or source.get('expert_id')!=job['expert_id'] or span is None:
            raise ValueError('Foreign transfer source')
        if (sid,span_id) in seen:raise ValueError('Duplicate transfer witness')
        seen.add((sid,span_id));owners.add(sid)
        start,end=span['start'],span['end']
        if not 0<=start<end<=len(source['text']) or source['text'][start:end]!=span['text']:
            raise ValueError('Changed transfer witness')
        citations.append(dict(evidence_id=sid,start=start,end=end,quote=span['text']))
    if len(owners)!=1:raise ValueError('One demonstrated capability must come from one paper')
    return dict(status='proposal_pending_audit',capability=proposal['capability'],target_operation=proposal['target_operation'],
                demand_citation=dict(start=target['start'],end=target['end'],quote=target['text']),citations=citations,
                mechanism=proposal['mechanism'],assumptions=proposal['assumptions'],remaining_demand_gaps=proposal['remaining_demand_gaps'],
                notice='Capability fact needs source audit; transfer is a conditional hypothesis, not prior target achievement or whole-project fit.')
