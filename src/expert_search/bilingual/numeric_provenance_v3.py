"""Post-hoc ambiguity hints for two narrow cross-language numeric expressions."""
from .constraints import numeric_anchors
from .numeric_provenance_v2 import audit_numeric_provenance_v2,possible_spaced_negative_values
import re


DIMENSION=re.compile(r'(?<![A-Za-z0-9])3\s*[- ]?\s*[dD](?![A-Za-z])')
HUNDRED_METERS=re.compile(r'(?<![A-Za-z0-9])100\s*(?:m|meters|metres)(?![A-Za-z])',re.I)


def audit_numeric_provenance_v3(parent,source_ids,query):
    """Classify possible format alternatives as uncertain, never as support.

    The two patterns were identified after inspecting flagged development cases.
    A matching phrase does not prove the same object, scope, unit or relation.
    No general Chinese numeral conversion is attempted.
    """
    base=audit_numeric_provenance_v2(parent,source_ids,query)
    observed={a['value'] for a in numeric_anchors(query)}
    known={a['value'] for a in numeric_anchors(parent)}|possible_spaced_negative_values(parent)
    absent=observed-known;alternatives=set()
    if '\u4e09\u7ef4' in parent and DIMENSION.search(query):alternatives.add('3')
    if '\u767e\u7c73' in parent and HUNDRED_METERS.search(query):alternatives.add('1E+2')
    ambiguous=absent&alternatives
    return {'still_absent_from_parent_v2':len(absent),'possible_cross_language_values':len(ambiguous),
            'still_absent_after_hints':len(absent-ambiguous),
            'scope':'Two post-hoc literal pattern hints only. Possible cross-language alternatives remain unjudged; do not infer semantic equivalence or change data status.'}
