"""Source-first planning helpers; retrieval scores never establish a label."""
from .source_counterfactual import normalized,locate,build_views


def paper_inventory(job):
    inventory={}
    for source in list(job['sources'].values())+job['reserve']:
        if source['expert_id']!=job['expert_id']:
            raise ValueError('Cross-expert planning source')
        if source['kind']=='paper':
            if source['id'] in inventory:raise ValueError('Duplicate planning source ID')
            inventory[source['id']]=source
    return inventory


def independent_inventory(job,plan):
    """Validate a joint hypothesis and remove its literal shadow copies first."""
    topics=plan.get('topics')
    relation=plan.get('relation')
    gaps=plan.get('original_demand_gaps')
    if (plan.get('feasible') is not True or not isinstance(topics,list) or len(topics)!=2 or
            any(not isinstance(t,str) or not t.strip() for t in topics) or
            any(normalized(a) in normalized(b) for a,b in ((topics[0],topics[1]),(topics[1],topics[0])))):
        raise ValueError('Require two distinct nonnested topics')
    if (not isinstance(relation,str) or not 10<=len(relation)<=240 or not isinstance(gaps,list) or
            not gaps or any(not isinstance(g,str) or not g.strip() for g in gaps)):
        raise ValueError('Missing scoped relationship or demand gaps')
    quote=plan.get('demand_quote')
    if not isinstance(quote,str) or not 20<=len(quote)<=400 or job['demand'].count(quote)!=1:
        raise ValueError('Invalid real-demand anchor')
    papers=paper_inventory(job)
    joint=locate(plan['joint_support'],papers)
    title=normalized(joint.get('title') or '')
    quote=normalized(plan['joint_support']['quote'])
    if len(title)<6 or len(quote)<20:raise ValueError('Require concrete joint witness')
    def alias(source):
        text=normalized(source['text'])
        return title in text or quote in text
    # Retain original context; do not hide a shadow copy in a profile field.
    if any(alias(s) for s in job['sources'].values() if s['kind']!='paper'):
        raise ValueError('Joint alias remains in nonpaper context')
    remaining={eid:s for eid,s in papers.items() if eid!=joint['id'] and not alias(s)}
    if len(remaining)<4:raise ValueError('Too few independent papers and replacement reserves')
    return joint,remaining


def assemble(job,plan,witnesses,allowed):
    """Bind exact independent quotes, then construct equal-count real controls."""
    joint,remaining=independent_inventory(job,plan)
    if not isinstance(witnesses,list) or len(witnesses)!=2:
        raise ValueError('Require two independent witnesses')
    if any(type(w.get('topic_index')) is not int for w in witnesses) or sorted(w['topic_index'] for w in witnesses)!=[0,1]:
        raise ValueError('Invalid topic witness indices')
    if not set(allowed)<=set(remaining):raise ValueError('Invalid independent retrieval pool')
    chosen={joint['id']:joint}
    for witness in witnesses:
        if witness.get('evidence_id') not in allowed:raise ValueError('Witness outside supplied independent pool')
        source=locate(witness,remaining);chosen[source['id']]=source
    order=sorted(remaining,key=lambda eid:(-job['selection']['paper_cosines'][eid],eid))
    for eid in order:
        if len(chosen)==4:break
        chosen[eid]=remaining[eid]
    fields={eid:s for eid,s in job['sources'].items() if s['kind']!='paper'}
    prepared=dict(job,sources={**fields,**chosen},reserve=[s for eid,s in remaining.items() if eid not in chosen])
    example=dict(plan,independent_support=witnesses)
    return prepared,example,build_views(prepared,example)
