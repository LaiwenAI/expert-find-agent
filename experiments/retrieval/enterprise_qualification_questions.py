"""Choose a clarification that can change verified candidate feasibility."""

from collections import Counter
from functools import lru_cache


def candidate_masks(profiles, fixed, disputed):
    masks = []
    for profile in profiles:
        if profile[0] and all(profile[i] for i in fixed):
            masks.append(sum(1 << slot for slot, i in enumerate(disputed) if not profile[i]))
    return masks


def decision_table(blockers, clauses, *, full_set=False):
    results = []
    for active in range(1 << clauses):
        feasible = sum(1 << i for i, missing in enumerate(blockers) if not active & missing)
        results.append(feasible if full_set else bool(feasible))
    return results


def ask(table):
    """Minimize worst-case questions, then mean under equal scenario weights."""

    @lru_cache(maxsize=None)
    def search(states, remaining):
        if remaining == 0 or len({table[world] for world in states}) == 1:
            return 0, 0, None
        choices = []
        for bit in range(remaining.bit_length()):
            flag = 1 << bit
            if not flag & remaining:
                continue
            yes = tuple(world for world in states if world & flag)
            no = tuple(world for world in states if not world & flag)
            yes_depth, yes_total, _ = search(yes, remaining ^ flag)
            no_depth, no_total, _ = search(no, remaining ^ flag)
            choices.append((1 + max(yes_depth, no_depth),
                            len(states) + yes_total + no_total, bit))
        return min(choices)

    size = len(table)
    if size < 1 or size & (size - 1):
        raise ValueError("Expected a complete binary interpretation table")
    return search(tuple(range(size)), size - 1)


def examine(profiles, baseline, alternate):
    fixed = set(baseline) & set(alternate)
    disputed = sorted(set(baseline) ^ set(alternate))
    blockers = candidate_masks(profiles, fixed, disputed)
    binary = decision_table(blockers, len(disputed))
    full = decision_table(blockers, len(disputed), full_set=True)
    depth, cost, first = ask(binary)
    full_depth, full_cost, _ = ask(full)
    resolved = binary[-1]
    results = Counter(
        requests=1, disputed_clauses=len(disputed), requests_with_dispute=int(bool(disputed)),
        invariant_availability=int(depth == 0), invariant_full_set=int(full_depth == 0),
        clarification_for_availability=int(depth > 0), clarification_for_full_set=int(full_depth > 0),
        single_question_always_decides_availability=int(depth == 1),
        single_question_always_decides_full_set=int(full_depth == 1),
        sum_worst_case_availability_questions=depth, sum_worst_case_full_set_questions=full_depth,
        possible_provisional_request=int(binary[0]),
        guaranteed_provisional_request=int(resolved),
    )
    # Equal weighting is a sensitivity scenario over possible answers,
    # not an estimate of how callers will resolve disputed clauses.
    values = {
        "uniform_mean_questions_availability": cost / len(binary),
        "uniform_mean_questions_full_set": full_cost / len(binary),
    }
    if first is not None:
        bit = 1 << first
        options = [[value for world, value in enumerate(binary) if bool(world & bit) == active]
                   for active in (False, True)]
        results.update(first_answer_resolves_some_branch=int(any(len(set(x)) == 1 for x in options)),
                       first_answer_resolves_both_branches=int(all(len(set(x)) == 1 for x in options)))
    return results, values
