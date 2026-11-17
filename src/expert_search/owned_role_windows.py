"""Optional source-role reordering with immutable, owned text windows."""
import hashlib

from .source_roles import normalize_roles,prioritize_roles


def segment_source(source,width=400):
    if not isinstance(source,dict) or not all(isinstance(source.get(k),str) and source[k] for k in ('id','expert_id','text')):
        raise ValueError('Require source ID, owner and text')
    if type(width)!=int or width<1:raise ValueError('Invalid window width')
    text=source['text']
    return [{'id':i,'evidence_id':source['id'],'expert_id':source['expert_id'],'start':start,
             'end':min(start+width,len(text)),'text':text[start:start+width]}
            for i,start in enumerate(range(0,len(text),width))]


def source_key(source,width):
    segment_source(source,width)
    return {'source_id':source['id'],'expert_id':source['expert_id'],'text_sha256':hashlib.sha256(source['text'].encode()).hexdigest(),'window_width':width}


def make_role_record(source,raw_roles,width=400):
    """Bind structurally normalized role hints to the exact owned source."""
    key=source_key(source,width);normalized=normalize_roles(raw_roles,len(segment_source(source,width)))
    return {'source_key':key,'roles':normalized['roles'],
            'normalization':{name:normalized[name] for name in ('missing','conflicts','within_role_duplicates')},
            'scope':'Structural normalization only; role correctness and attribution are unverified.'}


def select_owned_windows(source,ranking,k,role_record=None,width=400):
    """Reorder a complete baseline ranking; stale or invalid roles use baseline.

    Fixed-width windows are not sentences. Role labels are untrusted hints, not
    attribution or semantic judgments. A changed source or segmentation must
    not reuse its old role record.
    """
    windows=segment_source(source,width)
    if type(k)!=int or k<1 or not isinstance(ranking,(list,tuple)) or any(type(i)!=int for i in ranking) or sorted(ranking)!=list(range(len(windows))):
        raise ValueError('Require a complete unique window ranking and positive k')
    ordered=list(ranking);status='baseline_no_roles'
    if role_record is not None:
        try:
            if not isinstance(role_record,dict) or role_record.get('source_key')!=source_key(source,width):raise ValueError('Stale source')
            roles=normalize_roles(role_record.get('roles'),len(windows))['roles']
            ordered=prioritize_roles(ordered,roles);status='role_priority'
        except (TypeError,ValueError,KeyError):
            status='baseline_invalid_roles'
    return {'windows':[windows[i] for i in ordered[:k]],'status':status,
            'scope':'Owned raw text windows with exact offsets; role hints are unvalidated and may alter ranking. Missing or invalid roles use the baseline ordering. No expert capability or relevance claim.'}
