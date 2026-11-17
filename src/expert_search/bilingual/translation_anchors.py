"""Reversible numeric/comparator placeholders, not semantic translation proof.

Placeholder terminology control is inherited from constrained-translation work.
Only the explicitly supported comparator/unit grammar is canonicalized. Other
words stay in the model input and require semantic auditing after restoration.
"""
import re
import unicodedata
from .constraints import numeric_anchors, UNIT, UNITS
from .postvalidation import COMPARATORS, NEGATION

TOKEN = re.compile(r'\[\[NUM\d{6}\]\]')


def protect(text):
    if TOKEN.search(text):
        raise ValueError('Input contains reserved numeric placeholder')
    spans = []
    for index, anchor in enumerate(numeric_anchors(text)):
        start, end = anchor['start'], anchor['end']
        prefix = text[max(0, start - 100):start]
        operator = None
        for name, expression in COMPARATORS:
            match = re.search(r'(?<![A-Za-z])(?:' + expression + r')\s*$', prefix, re.I)
            if match and not NEGATION.search(re.split(r'[。；;!?！？\n,，]', prefix[:match.start()])[-1]):
                start -= len(prefix) - match.start()
                operator = name
                break
        unit_match = UNIT.match(text[end:])
        unit = None
        if unit_match:
            unit = UNITS[unit_match.group(1)]
            end += unit_match.end()
        postfix = re.match(r'\s*(以上|以下)', text[end:])
        # Contradictory prefix/postfix comparisons are left for the reviewer.
        if postfix and operator is None:
            operator = '>=' if postfix.group(1) == '以上' else '<='
            end += postfix.end()
        literal = unicodedata.normalize('NFKC', anchor['literal'])
        canonical = ((operator + ' ') if operator else '') + literal + ((' ' + unit) if unit else '')
        spans.append(dict(token='[[NUM' + str(index).zfill(6) + ']]', start=start, end=end, source=text[start:end], english=canonical))
    if any(a['end'] > b['start'] for a, b in zip(spans, spans[1:])):
        raise ValueError('Overlapping protected numeric expressions')
    output, position = [], 0
    for span in spans:
        output.extend([text[position:span['start']], span['token']])
        position = span['end']
    output.append(text[position:])
    return dict(masked=''.join(output), spans=spans)


def restore(translated, spans):
    expected = [s['token'] for s in spans]
    found = TOKEN.findall(translated)
    if len(found) != len(expected) or set(found) != set(expected):
        raise ValueError('Missing, duplicated or foreign numeric placeholder')
    for span in spans:
        translated = translated.replace(span['token'], span['english'])
    # Reject malformed remnants too; the model may not invent placeholder syntax.
    if '[[NUM' in translated:
        raise ValueError('Malformed numeric placeholder')
    return translated


def split_gaps(text):
    """Separate free-language gaps from fixed numeric blocks; no token copying."""
    spans = protect(text)['spans']
    gaps, position = [], 0
    for span in spans:
        gaps.append(text[position:span['start']])
        position = span['end']
    gaps.append(text[position:])
    return dict(gaps=gaps, spans=spans)


def assemble_gaps(translations, spans):
    if len(translations) != len(spans) + 1 or any(not isinstance(t, str) for t in translations):
        raise ValueError('Missing free-language gap')
    if any(numeric_anchors(t) or '[[NUM' in t for t in translations):
        raise ValueError('Free gap introduced a numeric expression')
    parts = []
    for index, gap in enumerate(translations):
        parts.append(gap)
        if index < len(spans):
            parts.append(spans[index]['english'])
    return ' '.join(p for p in parts if p).strip()
