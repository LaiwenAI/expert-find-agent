"""Versioned conservative comparator recheck; never rewrites frozen v1 outputs.

This is a deliberately limited grammar, not a semantic-equivalence certificate.
Unrecognized nearby negation makes the numeric check uncertain, even if both
languages have otherwise identical anchors. Units retain v1 case sensitivity.
"""
import re
import unicodedata
from .constraints import numeric_anchors, UNIT

COMPARATORS = (
    ('>=', r'not (?:below|under|lower than|less than)|no (?:less|lower) than|at least|不低于|不少于|不小于|大于等于|至少|≥|>='),
    ('<=', r'not (?:above|over|higher than|greater than|more than)|no (?:greater|higher|more) than|at most|up to|不超过|不高于|不大于|小于等于|至多|≤|<='),
    ('>', r'greater than|more than|higher than|over|above|大于|超过|高于|>'),
    ('<', r'less than|lower than|under|below|小于|低于|<'),
    ('+/-', r'plus or minus|±'),
)
NEGATION = re.compile(r'\b(?:not|no|never|without|neither)\b|不|未|非|无', re.I)


def checked_anchors(text):
    rows = numeric_anchors(text)
    for row in rows:
        prefix = unicodedata.normalize('NFKC', text[max(0, row['start'] - 100):row['start']])
        # A conservative clause scope. Negation in an earlier clause is irrelevant.
        prefix = re.split(r'[。；;!?！？\n,，]', prefix)[-1]
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
            if re.match(r'\s*(?:以上|以下)', postfix):
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
