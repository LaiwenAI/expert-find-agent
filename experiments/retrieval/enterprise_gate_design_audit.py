"""Design sensitivity for the frozen enterprise gate without opening any trial data."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

from experiments.enterprise_review_gate import assess, coverage_boundary, design, scenario_power
from experiments.retrieval.check_joint_review import sha

SIZES = (123, 250, 500, 1000, 2000)
ABSOLUTE = (
    (0.65, 0.01), (0.70, 0.01), (0.75, 0.00), (0.75, 0.01),
    (0.75, 0.02), (0.75, 0.03), (0.80, 0.01),
)
PAIRED = ((0.03, 0.20), (0.03, 0.40), (0.04, 0.30), (0.04, 0.50),
          (0.05, 0.30), (0.05, 0.50))


def absolute_limits(groups, floor, ceiling, alpha):
    start = coverage_boundary(groups, floor, alpha)
    if start is None:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    issued = np.arange(start, groups + 1)
    risk_limit = binom.ppf(alpha, issued, ceiling).astype(int)
    # The largest accepted count has a lower-tail probability at most alpha.
    risk_limit -= (binom.cdf(risk_limit, issued, ceiling) > alpha).astype(int)
    return issued, risk_limit


def absolute_pass_probability(groups, model, coverage, risk):
    issued, risk_limit = model
    if not len(issued):
        return 0.
    return float(np.dot(binom.pmf(issued, groups, coverage), binom.cdf(risk_limit, issued, risk)))


def passing_boundary(groups, k, margin, tails, *, inclusive=False):
    # Each changed group contributes plus or minus one supported slot.
    boundary = groups * k * margin
    cutoff = math.ceil(boundary - 1e-12) - 1 if inclusive else math.floor(boundary + 1e-12)
    draws = np.arange(groups + 1)
    lost = (draws + cutoff) // 2
    winning = np.zeros(groups + 1, dtype=int)
    for discordant in range(groups + 1):
        # Conditional on the number of nonzero bootstrap draws, count wins.
        weight = binom.pmf(draws, groups, discordant / groups)
        lo, hi = 0, discordant + 1
        while lo < hi:
            middle = (lo + hi) // 2
            probability = middle / discordant if discordant else 0.
            failure = np.dot(weight, binom.cdf(lost, draws, probability))
            if failure < tails:
                hi = middle
            else:
                lo = middle + 1
        winning[discordant] = lo
    return winning


def paired_pass_probability(groups, k, winning, gain, discordance):
    margin_slots = gain * k
    p_win_given_change = (1 + margin_slots / discordance) / 2
    if not 0 <= p_win_given_change <= 1:
        raise ValueError("Impossible gain and disagreement probability")
    changed = np.arange(groups + 1)
    return float(np.dot(binom.pmf(changed, groups, discordance),
                        binom.sf(winning - 1, changed, p_win_given_change)))


def current_gate(groups, floor, risk, confidence):
    alpha = (1 - confidence) / 2
    issued, allowed = absolute_limits(groups, floor, risk, alpha)
    boundaries = []
    for target in range(5):
        possible = issued[allowed >= target]
        boundaries.append(dict(
            adverse_or_unresolved=target,
            minimum_issued=int(possible[0]) if len(possible) else None,
        ))
    result = design(groups, floor, risk, confidence)
    for row in boundaries:
        if row["minimum_issued"] is None:
            continue
        n = row["minimum_issued"]
        count = row["adverse_or_unresolved"]
        assert assess(groups, n, count, 0, floor, risk, confidence)["passes"]
        assert not assess(groups, n-1, count, 0, floor, risk, confidence)["passes"]
    return dict(**result, boundaries=boundaries)


def run(protocol, output):
    spec = json.loads(protocol.read_bytes())
    floor, risk, confidence = (spec[key] for key in ("coverage_floor", "conditional_risk_ceiling", "confidence"))
    groups, k = spec["groups"], spec["top_k"]
    alpha = (1-confidence)/2
    sizes = sorted({groups, *SIZES})
    gates = current_gate(groups, floor, risk, confidence)
    absolute, paired = [], []
    for n in sizes:
        model = absolute_limits(n, floor, risk, alpha)
        absolute.extend(dict(groups=n, true_coverage=c, adverse_or_unresolved_given_issue=r,
                             power=absolute_pass_probability(n, model, c, r)) for c,r in ABSOLUTE)
        borders = {
            "superiority": passing_boundary(n, k, spec["minimum_grounded_p5_gain"], alpha),
            "noninferiority": passing_boundary(n, k, -spec["quality_noninferiority_margin"], alpha,
                                              inclusive=True),
        }
        for name, gain, discordance in (
            *[("superiority", gain, discordance) for gain, discordance in PAIRED],
            ("noninferiority", 0., .20), ("noninferiority", 0., .40),
        ):
            paired.append(dict(groups=n, endpoint=name, true_p5_difference=gain, discordance=discordance,
                               power=paired_pass_probability(n, k, borders[name], gain, discordance)))
    # Agreement with the deterministic scorer's original planning helper.
    for row in absolute:
        if row["groups"] != groups:
            continue
        expected = scenario_power(groups, floor, risk, confidence, row["true_coverage"],
                                  row["adverse_or_unresolved_given_issue"])
        if not math.isclose(row["power"], expected, abs_tol=1e-10):
            raise ValueError("Design calculation disagrees with the frozen gate")
    with output.open("x", encoding="utf-8") as f:
        json.dump(dict(
            protocol_sha256=sha(protocol), code_sha256=sha(Path(__file__)), gates=gates,
            absolute_single_arm=absolute, paired_slots=paired,
            assumptions=dict(iid_groups=True, paired_slots_support=[-1, 0, 1],
                             bootstrap="Exact empirical resampling CDF; no finite-draw simulation",
                             absolute_risk="Confirmed adverse or still unresolved among issued groups"),
        ), f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("protocol", type=Path)
    args.add_argument("output", type=Path)
    values = args.parse_args()
    run(values.protocol, values.output)
