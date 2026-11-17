"""Explicit conjunction of evidence-scoped expertise predicates, no training."""
CHOICES = [dict(value='demonstrated', description='The supplied candidate publication record demonstrates the stated requirement.'), dict(value='not_demonstrated', description='The supplied record does not demonstrate this requirement; this is not a claim about actual inability.'), dict(value='uncertain', description='The supplied text does not allow a determinate evidence-scoped judgment.')]


def questions(criterion):
    return [dict(type='choice', name='core', instructions=criterion + '\nFactor A only: Does the supplied candidate record demonstrate at least ONE CORE expertise requirement of the target paper? CORE is an underlying central problem/method, not a shared incidental topic or a requirement to know every core area. Do not answer the full conjunction here.', choices=CHOICES),
            dict(type='choice', name='breadth', instructions=criterion + '\nFactor B only: Does the supplied candidate record demonstrate at least TWO DISTINCT relevant expertise requirements of the target paper? A requirement can be CORE or Secondary. Two phrases for the same skill, or broad incidental vocabulary, do not count as distinct requirements. This factor alone does not require a core match; Factor A checks that separately.', choices=CHOICES)]


def combine(labels):
    allowed = {'demonstrated', 'not_demonstrated', 'uncertain'}
    if set(labels) != {'core', 'breadth'} or any(value not in allowed for value in labels.values()):
        raise ValueError('Invalid factored suitability answers')
    if 'not_demonstrated' in labels.values():
        return 0
    if all(value == 'demonstrated' for value in labels.values()):
        return 1
    return None
