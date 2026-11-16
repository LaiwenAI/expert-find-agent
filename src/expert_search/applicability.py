"""Evidence-grounded applicability ranking, separate from factual entailment.

Ordinal protocol scores are not probabilities of solving an enterprise project.
"""
import math

LEVELS=('direct_application','transferable_method','topic_only','insufficient')


def assess_candidate(expert_id,requirements,assessments,sources):
    if not requirements:raise ValueError('No enterprise requirements')
    if set(assessments)-set(requirements):raise ValueError('Unexpected requirement assessment')
    total=0.;direct=0.;transfer=0.;topic=0.;unresolved=[];records={}
    for rid,requirement in requirements.items():
        weight=requirement.get('weight',1.)
        if type(weight) not in (float,int) or not math.isfinite(weight) or weight<=0:raise ValueError('Invalid requirement weight')
        total+=weight
        a=assessments.get(rid,dict(level='insufficient',citations=[],assumptions=[],transfer_reason=None))
        level=a.get('level')
        if level not in LEVELS:raise ValueError('Invalid applicability level')
        citations=a.get('citations',[])
        if not isinstance(citations,list) or (level!='insufficient' and not citations):raise ValueError('Evidence required for applicability')
        resolved=[]
        for citation in citations:
            sid=citation.get('evidence_id');source=sources.get(sid)
            if source is None or source.get('expert_id')!=expert_id:raise ValueError('Invalid evidence owner')
            start,end=citation.get('start'),citation.get('end')
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(source['text']):raise ValueError('Invalid evidence span')
            if citation.get('quote')!=source['text'][start:end]:raise ValueError('Evidence span does not match')
            resolved.append(citation)
        assumptions=a.get('assumptions',[])
        if not isinstance(assumptions,list) or any(not isinstance(x,str) or not x.strip() for x in assumptions):raise ValueError('Invalid transfer assumptions')
        if level=='transferable_method':
            if not isinstance(a.get('transfer_reason'),str) or not a['transfer_reason'].strip() or not assumptions:raise ValueError('Transfer requires mechanism and unresolved assumptions')
            transfer+=weight
        elif level=='direct_application':direct+=weight
        elif level=='topic_only':topic+=weight
        if level in ('topic_only','insufficient') or assumptions:unresolved.append(rid)
        records[rid]=dict(a,citations=resolved)
    return dict(expert_id=expert_id,direct_coverage=direct/total,transfer_coverage=transfer/total,
        actionable_coverage=(direct+transfer)/total,topic_coverage=topic/total,unresolved_requirements=unresolved,assessments=records,
        notice='Model applicability and cited-span existence need independent semantic review. Coverage is not project success probability; future performance is not an established achievement.')


def rank_candidates(candidates):
    return sorted(candidates,key=lambda r:(-r['actionable_coverage'],-r['direct_coverage'],-r['topic_coverage'],r['expert_id']))
