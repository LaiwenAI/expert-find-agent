"""Validate demand-anchored operations and separate system obligations."""
import re


def validate_plan(demand, references, plan):
    operations, obligations = plan.get('operations'), plan.get('obligations')
    if not isinstance(operations, list) or not 1 <= len(operations) <= 24:
        raise ValueError('Invalid operation count')
    if not isinstance(obligations, list) or len(obligations) > 12 or plan.get('complete') is not True:
        raise ValueError('Incomplete operation plan')
    ids, signatures = set(), set()
    for row in operations+obligations:
        rid = row.get('id')
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError('Duplicate or invalid plan ID')
        ids.add(rid)
        anchors = row.get('span_ids')
        if not isinstance(anchors, list) or not anchors or len(set(anchors)) != len(anchors):
            raise ValueError('Invalid demand anchors')
        for sid in anchors:
            if sid not in references:
                raise ValueError('Unknown demand anchor')
            span = references[sid]
            if demand[span['start']:span['end']] != span['text']:
                raise ValueError('Changed demand anchor')
    for row in operations:
        for field in ('object', 'operation'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError('Missing operation or object')
        qualifiers = row.get('qualifiers')
        if not isinstance(qualifiers, list) or any(not isinstance(q, str) or not q.strip() for q in qualifiers):
            raise ValueError('Invalid operation qualifiers')
        normalize = lambda text: re.sub(r'\s+', '', text).casefold()
        signature = (normalize(row['object']), normalize(row['operation']), tuple(sorted(normalize(q) for q in qualifiers)))
        if signature in signatures:
            raise ValueError('Exact duplicate operation with same scope')
        signatures.add(signature)
    operation_ids = {r['id'] for r in operations}
    for row in obligations:
        if row.get('kind') not in ('integration', 'validation', 'deployment') or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Invalid system obligation')
        dependencies = row.get('operation_ids')
        if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies) or any(i not in operation_ids for i in dependencies):
            raise ValueError('Invalid obligation dependencies')
    return dict(operations=operations, obligations=obligations,
                notice='Structural and literal-anchor validation only. Semantic atomicity, missing operations and duplicate meanings require separate review. System obligations are not implied by component evidence.')
