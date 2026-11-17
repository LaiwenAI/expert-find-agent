"""Flag numeric values absent from cited or full source, without judging meaning."""
from .constraints import lossless_segments,numeric_anchors


def audit_numeric_provenance(parent,source_ids,query):
    """Compare detected normalized numeric values to cited and full parent text.

    This is a high-recall diagnostic, not a grounding or hallucination label.
    Numeric extraction can confuse identifiers and quantities, miss spelled-out
    numbers, and cannot check object, unit, comparator, arithmetic or context.
    Citations may omit relevant spans, so values present elsewhere are separate
    from values absent from the whole source. Do not promote a query to gold.
    """
    if not isinstance(parent,str) or not parent or not isinstance(query,str) or not isinstance(source_ids,list) or not source_ids or any(not isinstance(x,str) for x in source_ids) or len(set(source_ids))!=len(source_ids):
        raise ValueError('Invalid parent, query or citation IDs')
    segments={s['id']:s['zh'] for s in lossless_segments(parent)}
    if set(source_ids)-set(segments):raise ValueError('Foreign source segment')
    values=lambda text:{a['value'] for a in numeric_anchors(text)}
    original=values(parent);cited=set().union(*(values(segments[x]) for x in source_ids));observed=values(query)
    return {'detected_query_values':len(observed),'in_cited_values':len(observed&cited),
            'elsewhere_in_parent_values':len((observed-cited)&original),'absent_from_parent_values':len(observed-original),
            'scope':'Normalized values only; extraction misses cases and does not establish semantic grounding, unit equivalence or privacy.'}
