from math import comb
import unittest

from experiments.enterprise_review_gate import assess, design, scenario_power, upper_bound


class EnterpriseReviewGateTest(unittest.TestCase):
    def test_zero_failure_limit_and_simultaneous_requirements(self):
        self.assertAlmostEqual(upper_bound(0, 123, .025), 1 - .025 ** (1 / 123))
        planned = design(123, .60, .05)
        self.assertEqual(planned["minimum_issued_if_zero_adverse"], 72)
        self.assertEqual(planned["minimum_issued_for_joint_gate_if_zero_adverse"], 85)
        self.assertEqual(planned["allowable_adverse_or_unresolved_if_all_issue"], 1)

    def test_unresolved_issued_group_counts_against_risk(self):
        self.assertTrue(assess(123, 123, 1, 0, .60, .05)["passes"])
        self.assertFalse(assess(123, 123, 1, 1, .60, .05)["passes"])
        self.assertFalse(assess(123, 0, 0, 0, .60, .05)["passes"])

    def test_prospective_power_matches_explicit_joint_events(self):
        groups = 28
        params = dict(coverage_floor=.35, risk_ceiling=.20, confidence=.90)
        exact = sum(
            comb(groups, issued) * .6 ** issued * .4 ** (groups - issued)
            * comb(issued, events) * .07 ** events * .93 ** (issued - events)
            for issued in range(groups + 1)
            for events in range(issued + 1)
            if assess(groups, issued, events, 0, **params)["passes"]
        )
        self.assertAlmostEqual(scenario_power(groups, **params, actual_coverage=.6, actual_risk=.07), exact)
