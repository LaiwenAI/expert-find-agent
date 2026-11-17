"""Optional fixed-threshold Decisions-to-reasoning policy.

The native probability is an uncalibrated routing score, not a correctness guarantee.
No fallback is performed on a failed request; failures remain unresolved.
"""
import math
import time
from .choice_judgments import decide


def selective_decision(client, records, question, slow_decision, threshold):
    """Run one native decision and escalate below a fixed maximum-choice score.

    ``slow_decision`` receives the same records and question and must return one
    allowed choice value. Client setup and any later evidence checks are excluded.
    Callers must validate the policy on task-specific held-out judgments.
    """
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0, 1]')
    choices = question.get('choices', [])
    allowed = {choice['value'] for choice in choices}
    if len(allowed) < 2 or len(allowed) != len(choices) or not callable(slow_decision):
        raise ValueError('Require distinct choices and a callable slow decision')
    started = time.perf_counter()
    result = dict(status='unresolved', label=None, escalated=False, maximum_choice_probability=None,
                  failure_stage=None, error_type=None)
    try:
        native = decide(client, records, [question])
        answer = native['response']['answers'][0]
        score = max(p['probability'] for p in answer['probabilities'])
        result['maximum_choice_probability'] = score
        result['escalated'] = score < threshold
        if result['escalated']:
            result['failure_stage'] = 'slow'
            label = slow_decision(records, question)
            if label not in allowed:
                raise ValueError('Slow decision returned a foreign choice')
        else:
            label = native['labels'][question['name']]
        result.update(status='valid', label=label, failure_stage=None)
    except Exception as error:
        result['failure_stage'] = result['failure_stage'] or 'native'
        result['error_type'] = type(error).__name__
    result['seconds'] = time.perf_counter() - started
    return result
