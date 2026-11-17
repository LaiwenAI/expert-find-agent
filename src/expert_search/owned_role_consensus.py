"""Conservative agreement of source-bound role hints."""
from .owned_role_windows import make_role_record,segment_source,source_key
from .source_roles import ROLES,normalize_roles


def agree_role_records(source,first,second,width=400):
    """Retain shared role labels; mark every disagreement uncertain."""
    key=source_key(source,width);size=len(segment_source(source,width));assignments=[]
    for record in (first,second):
        if not isinstance(record,dict) or record.get('source_key')!=key:raise ValueError('Stale or missing source role record')
        roles=normalize_roles(record.get('roles'),size)['roles']
        assignments.append({i:role for role,ids in roles.items() for i in ids})
    combined={role:[] for role in ROLES};disagreements=0
    for i in range(size):
        a,b=assignments[0][i],assignments[1][i]
        if a!=b:disagreements+=1
        combined[a if a==b else 'uncertain'].append(i)
    result=make_role_record(source,combined,width);result['disagreements']=disagreements
    result['scope']='Agreement is a conservative consistency rule, not independent semantic verification or calibrated confidence.'
    return result
