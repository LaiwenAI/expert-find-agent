"""Fixed reciprocal ranks, with source ownership and explicit score ties."""
import math


def reciprocal_scores(values, constant=60, positive_only=False):
    if not math.isfinite(constant) or constant<=0 or any(not math.isfinite(v) for v in values.values()):
        raise ValueError('Invalid reciprocal-rank inputs')
    order=sorted((i for i,v in values.items() if not positive_only or v>0),key=lambda i:(-values[i],i))
    result={};previous=None;rank=0
    for position,key in enumerate(order,1):
        if previous is None or values[key]!=previous:rank=position
        result[key]=1/(constant+rank);previous=values[key]
    return result


def source_fusion(semantic,lexical,profiles,constant=60):
    dense=reciprocal_scores(semantic,constant)
    sparse=reciprocal_scores({i:lexical.get(i,0.) for i in semantic},constant,positive_only=True)
    combined={i:dense[i]+sparse.get(i,0.) for i in semantic};result={}
    for eid,ids in profiles.items():
        ids=set(ids)
        if ids-set(combined):raise ValueError('Missing owned source vector')
        top=sorted(ids,key=lambda i:(-combined[i],i))[:3]
        result[eid]=dict(source_max=combined[top[0]] if top else 0.,source_top3=sum(combined[i] for i in top)/3,witnesses=top)
    return result
