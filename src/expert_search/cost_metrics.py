"""Cost-per-correctness baseline from Xie et al., EACL 2026."""
import math


def tokens_per_correctness(records, input_weight=0.25, search_weight=500.0):
    """Return weighted token/search cost per correct response; infinity if none.

    Correctness must be supplied by the evaluator, including the appropriate
    abstention judgment for unanswerable cases. Missing judgments are rejected.
    """
    if not records or not math.isfinite(input_weight) or not math.isfinite(search_weight) or input_weight<0 or search_weight<0:
        raise ValueError('Invalid records or cost weights')
    cost=0.0;correct=0
    for row in records:
        if type(row.get('correct')) is not bool:
            raise ValueError('Every response needs a correctness judgment')
        values=[row.get(k) for k in ('output_tokens','input_tokens','search_calls')]
        if any(type(v) is not int or v<0 for v in values):
            raise ValueError('Cost counts must be nonnegative integers')
        output_tokens,input_tokens,search_calls=values
        cost+=output_tokens+input_weight*input_tokens+search_weight*search_calls
        correct+=row['correct']
    return cost/correct if correct else math.inf


def incomplete_tokens_per_correctness_bounds(records, input_weight=0.25, search_weight=500.0):
    """Sharp box bounds when unknown correctness may independently be true or false."""
    if not records:
        raise ValueError('At least one attempt is required')
    if any(row.get('correct') is not None and type(row.get('correct')) is not bool for row in records):
        raise ValueError('Correctness must be true, false, or unknown')
    optimistic=[dict(row,correct=True if row.get('correct') is None else row['correct']) for row in records]
    pessimistic=[dict(row,correct=False if row.get('correct') is None else row['correct']) for row in records]
    return (tokens_per_correctness(optimistic,input_weight,search_weight),
            tokens_per_correctness(pessimistic,input_weight,search_weight))
