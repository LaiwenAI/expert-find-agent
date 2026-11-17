"""Versioned conservative comparator recheck; never rewrites frozen v1 outputs.

This is a deliberately limited grammar, not a semantic-equivalence certificate.
Unrecognized nearby negation makes the numeric check uncertain, even if both
languages have otherwise identical anchors. Units retain v1 case sensitivity.
"""
import re
import unicodedata
from .constraints import numeric_anchors, UNIT

COMPARATORS = (
    ('>=', 'not (?:below|under|lower than|less than)|no (?:less|lower) than|at least|\u4e0d\u4f4e\u4e8e|\u4e0d\u5c11\u4e8e|\u4e0d\u5c0f\u4e8e|\u5927\u4e8e\u7b49\u4e8e|\u81f3\u5c11|\u2265|>='),
    ('<=', 'not (?:above|over|higher than|greater than|more than)|no (?:greater|higher|more) than|at most|up to|\u4e0d\u8d85\u8fc7|\u4e0d\u9ad8\u4e8e|\u4e0d\u5927\u4e8e|\u5c0f\u4e8e\u7b49\u4e8e|\u81f3\u591a|\u2264|<='),
    ('>', 'greater than|more than|higher than|over|above|\u5927\u4e8e|\u8d85\u8fc7|\u9ad8\u4e8e|>'),
    ('<', 'less than|lower than|under|below|\u5c0f\u4e8e|\u4f4e\u4e8e|<'),
    ('+/-', r'plus or minus|±'),
)
NEGATION = re.compile('\\b(?:not|no|never|without|neither)\\b|\u4e0d|\u672a|\u975e|\u65e0', re.I)


def checked_anchors(text):
    rows = numeric_anchors(text)
    for row in rows:
        prefix = unicodedata.normalize('NFKC', text[max(0, row['start'] - 100):row['start']])
        # A conservative clause scope. Negation in an earlier clause is irrelevant.
        prefix = re.split('[\u3002\uff1b;!?\uff01\uff1f\\n,\uff0c]', prefix)[-1]
        operator = None
        residue = prefix
        for name, expression in COMPARATORS:
            match = re.search(r'(?<![A-Za-z])(?:' + expression + r')\s*$', prefix, re.I)
            if match:
                operator = name
                residue = prefix[:match.start()]
                break
        # Preserve the v1 Chinese postfix comparator when no prefix was found.
        if operator is None and row['operator'] in ('>=', '<='):
            suffix = unicodedata.normalize('NFKC', text[row['end']:row['end'] + 30])
            unit = UNIT.match(suffix)
            postfix = suffix[unit.end():] if unit else suffix
            if re.match('\\s*(?:\u4ee5\u4e0a|\u4ee5\u4e0b)', postfix):
                operator = row['operator']
        row['operator'] = operator
        row['uncertain_negation'] = bool(NEGATION.search(residue))
    return rows


def check_pair(zh, en):
    source, target = checked_anchors(zh), checked_anchors(en)
    signature = lambda rows: [(r['value'], r['unit'], r['operator']) for r in rows]
    uncertain = any(r['uncertain_negation'] for r in source + target)
    match = signature(source) == signature(target)
    return dict(version='numeric-postcheck-v2', status='uncertain' if uncertain else ('match' if match else 'mismatch'), source_anchors=source, target_anchors=target,
                scope='Limited numeric/comparator check only; matching does not establish grounding, object scope, modality, translation equivalence, or relevance. Unsupported nearby negation is uncertain. No unit conversion or numeric-free semantic check.')
