"""Plan simultaneous coverage and conditional evidence-risk bounds at group level."""

import argparse
import json
import math


def binomial_lower_tail(events, groups, estimate):
    if estimate == 0 or events == groups:
        return 1.
    if estimate == 1:
        return 0.
    log_p = math.log(estimate)
    log_q = math.log1p(-estimate)
    total = math.lgamma(groups + 1)
    return math.fsum(
        math.exp(total - math.lgamma(i + 1) - math.lgamma(groups - i + 1)
                 + i * log_p + (groups - i) * log_q)
        for i in range(events + 1)
    )


def upper_bound(events, groups, alpha):
    if not 0 <= events <= groups or not 0 < alpha < 1:
        raise ValueError("Invalid event count or confidence level")
    if events == groups:
        return 1.
    low, high = 0., 1.
    for _ in range(58):
        middle = (low + high) / 2
        if binomial_lower_tail(events, groups, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def check_settings(groups, coverage_floor, risk_ceiling, confidence):
    if groups < 1 or not all(0 < value < 1 for value in (coverage_floor, risk_ceiling, confidence)):
        raise ValueError("Require positive groups and probability settings in (0, 1)")


def coverage_boundary(groups, floor, alpha):
    low, high = 1, groups + 1
    while low < high:
        issued = (low + high) // 2
        if binomial_lower_tail(groups - issued, groups, 1 - floor) <= alpha:
            high = issued
        else:
            low = issued + 1
    return None if low > groups else low


def risk_allowance(issued, ceiling, alpha):
    low, high = -1, issued
    while low + 1 < high:
        events = (low + high) // 2
        if binomial_lower_tail(events, issued, ceiling) <= alpha:
            low = events
        else:
            high = events
    return None if low < 0 else low


def assess(groups, issued, adverse, unresolved, coverage_floor, risk_ceiling, confidence=.95):
    """Use group-level counts, treating every unresolved issued group as adverse."""
    check_settings(groups, coverage_floor, risk_ceiling, confidence)
    if not 0 <= adverse <= adverse + unresolved <= issued <= groups:
        raise ValueError("Use disjoint adverse and unresolved counts among issued groups")
    alpha = (1. - confidence) / 2
    coverage = 1. - upper_bound(groups - issued, groups, alpha)
    upper_risk = upper_bound(adverse + unresolved, issued, alpha)
    return dict(
        groups=groups, issued=issued, confirmed_adverse=adverse, unresolved=unresolved,
        coverage_lower=coverage, conditional_risk_upper=upper_risk,
        simultaneous_confidence=confidence, per_endpoint_alpha=alpha,
        passes=coverage >= coverage_floor and upper_risk <= risk_ceiling,
    )


def design(groups, coverage_floor, risk_ceiling, confidence=.95):
    check_settings(groups, coverage_floor, risk_ceiling, confidence)
    alpha = (1. - confidence) / 2
    smallest_coverage_count = coverage_boundary(groups, coverage_floor, alpha)
    zero_adverse_count = math.ceil(math.log(alpha) / math.log1p(-risk_ceiling))
    smallest_joint_count = None if smallest_coverage_count is None else max(smallest_coverage_count, zero_adverse_count)
    if smallest_joint_count is not None and smallest_joint_count > groups:
        smallest_joint_count = None
    maximum_adverse = risk_allowance(groups, risk_ceiling, alpha)
    return dict(
        groups=groups, coverage_floor=coverage_floor, risk_ceiling=risk_ceiling,
        simultaneous_confidence=confidence, minimum_issued_for_coverage=smallest_coverage_count,
        minimum_issued_if_zero_adverse=zero_adverse_count, minimum_issued_for_joint_gate_if_zero_adverse=smallest_joint_count,
        allowable_adverse_or_unresolved_if_all_issue=maximum_adverse,
    )


def scenario_power(groups, coverage_floor, risk_ceiling, confidence, actual_coverage, actual_risk):
    """Calculate prospective power under independent groups and specified event probabilities."""
    check_settings(groups, coverage_floor, risk_ceiling, confidence)
    if not 0 < actual_coverage < 1 or not 0 <= actual_risk <= 1:
        raise ValueError("Invalid scenario probabilities")
    alpha = (1. - confidence) / 2
    coverage_minimum = coverage_boundary(groups, coverage_floor, alpha)
    if coverage_minimum is None:
        return 0.
    log_yes, log_no = math.log(actual_coverage), math.log1p(-actual_coverage)
    total = math.lgamma(groups + 1)
    probability = []
    for issued in range(coverage_minimum, groups + 1):
        maximum = risk_allowance(issued, risk_ceiling, alpha)
        if maximum is not None:
            mass = math.exp(total - math.lgamma(issued + 1) - math.lgamma(groups - issued + 1)
                            + issued * log_yes + (groups - issued) * log_no)
            probability.append(mass * binomial_lower_tail(maximum, issued, actual_risk))
    return math.fsum(probability)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", type=int)
    parser.add_argument("--coverage-floor", type=float, required=True)
    parser.add_argument("--risk-ceiling", type=float, required=True)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--issued", type=int)
    parser.add_argument("--adverse", type=int)
    parser.add_argument("--unresolved", type=int)
    parser.add_argument("--actual-coverage", type=float)
    parser.add_argument("--actual-risk", type=float)
    args = vars(parser.parse_args())
    scenario = {key: args.pop(key) for key in ("actual_coverage", "actual_risk")}
    if any(value is None for value in scenario.values()) and any(value is not None for value in scenario.values()):
        parser.error("Supply both scenario probabilities")
    if args["issued"] is None:
        if args["adverse"] is not None or args["unresolved"] is not None:
            parser.error("An outcome assessment requires all three observed counts")
        result = design(**{key: value for key, value in args.items() if value is not None})
        if scenario["actual_coverage"] is not None:
            result["scenario"] = scenario | {"power": scenario_power(
                **{key: args[key] for key in ("groups", "coverage_floor", "risk_ceiling", "confidence")}, **scenario,
            )}
    else:
        if args["adverse"] is None or args["unresolved"] is None:
            parser.error("Supply both adverse and unresolved counts explicitly")
        if scenario["actual_coverage"] is not None:
            parser.error("Scenario probabilities apply to prospective design only")
        result = assess(**args)
    print(json.dumps(result, indent=2))
