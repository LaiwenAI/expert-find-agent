"""Estimate joint pass probabilities in a prospective paired enterprise trial."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.retrieval.check_joint_review import sha
from experiments.retrieval.enterprise_gate_design_audit import (
    absolute_limits,
    absolute_pass_probability,
    paired_pass_probability,
    passing_boundary,
)

SIZES = (355, 500, 750, 1000, 1500, 2000)
COVERAGE = .75
REPEATS = 100_000
SEED = 20261121
# States are abstained, issued with adverse or unresolved evidence, and clean.
STATES = tuple((a, b, int(a == 2) - int(b == 2))
               for a in range(3) for b in range(3)) + ((2, 2, -1), (2, 2, 1))
ARM_A, ARM_B, CHANGE = np.asarray(STATES).T


def joint_distribution(risk, shared_fraction, gain, discordance, top_k):
    """Couple issuance and errors, then distribute improvements among clean pairs."""
    marginal = np.array([1 - COVERAGE, COVERAGE * risk, COVERAGE * (1 - risk)])
    pair = shared_fraction * np.diag(marginal) + (1 - shared_fraction) * np.outer(marginal, marginal)
    baseline_discordance = 2 * pair[2, :2].sum()
    extra_discordance = discordance - baseline_discordance
    slot_gain = gain * top_k
    new_loss = (extra_discordance - slot_gain) / 2
    new_win = (extra_discordance + slot_gain) / 2
    pair[2, 2] -= extra_discordance
    probabilities = np.append(pair.ravel(), [new_loss, new_win])
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Incompatible coverage, dependence, gain, and discordance")
    assert np.isclose(probabilities @ CHANGE, slot_gain)
    assert np.isclose(probabilities @ (CHANGE != 0), discordance)
    return probabilities


def evaluate(size, distribution, absolute, paired, rng):
    counts = rng.multinomial(size, distribution, size=REPEATS)
    issued, allowed = absolute
    max_adverse_by_issued = np.full(size + 1, -1)
    max_adverse_by_issued[issued] = allowed
    a = counts @ (ARM_A == 1) <= max_adverse_by_issued[counts @ (ARM_A > 0)]
    b = counts @ (ARM_B == 1) <= max_adverse_by_issued[counts @ (ARM_B > 0)]
    changed = counts @ (CHANGE != 0)
    quality = counts @ (CHANGE > 0) >= paired[changed]
    return {name: float(passed.mean()) for name, passed in (
        ("first_arm", a), ("second_arm", b), ("paired_quality", quality),
        ("both_absolute_gates", a & b), ("all_gates", a & b & quality),
    )}


def run(protocol, output):
    spec = json.loads(protocol.read_bytes())
    alpha = (1 - spec["confidence"]) / 2
    k = spec["top_k"]
    rng = np.random.default_rng(SEED)
    results = []
    for size in SIZES:
        absolute = absolute_limits(size, spec["coverage_floor"], spec["conditional_risk_ceiling"], alpha)
        for endpoint, gain, margin, inclusive in (
            ("superiority", .03, spec["minimum_grounded_p5_gain"], False),
            ("noninferiority", 0., -spec["quality_noninferiority_margin"], True),
        ):
            paired = passing_boundary(size, k, margin, alpha, inclusive=inclusive)
            for risk in (.01, .02, .03):
                absolute_exact = absolute_pass_probability(size, absolute, COVERAGE, risk)
                for shared in (.9, 1.):
                    for discordance in (.20, .40):
                        distribution = joint_distribution(risk, shared, gain, discordance, k)
                        measured = evaluate(size, distribution, absolute, paired, rng)
                        paired_exact = paired_pass_probability(size, k, paired, gain, discordance)
                        error = max(abs(measured[key] - target) for key, target in (
                            ("first_arm", absolute_exact), ("second_arm", absolute_exact),
                            ("paired_quality", paired_exact),
                        ))
                        if error > .009:
                            raise ValueError("Simulated marginals disagree with exact planning probabilities")
                        results.append(dict(
                            groups=size, endpoint=endpoint, risk=risk, shared_state_fraction=shared,
                            true_p5_gain=gain, discordance=discordance, **measured,
                            marginal_probability_upper_bound=min(absolute_exact, paired_exact),
                            marginal_probability_lower_bound=max(0., 2 * absolute_exact + paired_exact - 2),
                        ))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(dict(protocol_sha256=sha(protocol), code_sha256=sha(Path(__file__)), seed=SEED,
                       repetitions=REPEATS, coverage=COVERAGE, scenarios=results), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("protocol", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
