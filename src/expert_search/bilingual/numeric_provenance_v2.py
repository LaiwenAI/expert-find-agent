"""Conservative spaced-sign alternatives for numeric provenance diagnostics."""
from decimal import Decimal
import re
import unicodedata
from .constraints import lossless_segments,numeric_anchors


SPACED_SIGN=re.compile(r'(?<![A-Za-z0-9.])[-\u2212]\s+\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')


def possible_spaced_negative_values(text):
    """Return alternatives, never a decision that a dash means negative.

    A spaced dash can be a minus sign, range marker, bullet or punctuation.
    Every match is therefore reported as ambiguous and must be reviewed.
    """
    normalized=unicodedata.normalize('NFKC',text)
    return {str(Decimal(re.sub(r'\s+','',m.group()).replace(',','').replace('\u2212','-')).normalize()) for m in SPACED_SIGN.finditer(normalized)}


def audit_numeric_provenance_v2(parent,source_ids,query):
    if not isinstance(parent,str) or not parent or not isinstance(query,str) or not isinstance(source_ids,list) or not source_ids or any(not isinstance(x,str) for x in source_ids) or len(set(source_ids))!=len(source_ids):raise ValueError('Invalid parent, query or citation IDs')
    segments={s['id']:s['zh'] for s in lossless_segments(parent)}
    if set(source_ids)-set(segments):raise ValueError('Foreign source segment')
    values=lambda text:{a['value'] for a in numeric_anchors(text)}
    parent_values=values(parent);query_values=values(query);cited=set().union(*(values(segments[x]) for x in source_ids))
    parent_alternatives=possible_spaced_negative_values(parent)
    cited_alternatives=set().union(*(possible_spaced_negative_values(segments[x]) for x in source_ids))
    return {'detected_query_values':len(query_values),'absent_from_parent_v1':len(query_values-parent_values),
            'ambiguous_parent_spaced_sign_values':len((query_values-parent_values)&parent_alternatives),
            'still_absent_from_parent_values':len(query_values-parent_values-parent_alternatives),
            'absent_from_cited_v1':len(query_values-cited),
            'ambiguous_cited_spaced_sign_values':len((query_values-cited)&cited_alternatives),
            'still_absent_from_cited_values':len(query_values-cited-cited_alternatives),
            'scope':'Possible spaced-sign alternatives are ambiguous, not verified negatives or grounding. Values only; semantic and privacy checks remain necessary.'}
