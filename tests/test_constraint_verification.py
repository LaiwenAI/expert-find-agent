import random
import unittest

from expert_search.constraint_verification import resolve_constraints, verify_request
from expert_search.requirement_packets import clauses


class ConstraintVerificationTest(unittest.TestCase):
    def test_preserves_first_witness_even_when_later_bundle_supports_everything(self):
        checks = {(0, "early"): {0: True}, (0, "later"): {0: True, 1: True}}
        result = resolve_constraints([(0, (("early", "later"), ("later",)))],
                                     lambda e, s: checks[e, s], top=1)
        self.assertEqual(result["ranking"], [(0, ("early", "later"))])

    def test_unavailable_role_does_not_promote_expert(self):
        rows = [(0, (("paper",), ("paper",))), (1, (("backup",), ("backup",)))]
        decisions = {(0, "paper"): {0: True, 1: None}, (1, "backup"): {0: True, 1: True}}
        result = resolve_constraints(rows, lambda e, s: decisions[e, s], top=1)
        self.assertEqual(result["ranking"], [(1, ("backup", "backup"))])

    def test_all_strategies_match_random_exhaustive_replays(self):
        rng = random.Random(20261119)
        for _ in range(300):
            pool = [(e, tuple(tuple(rng.sample(range(4), rng.randrange(5))) for _ in range(3)))
                    for e in range(5)]
            scores = {(e, s): {i: rng.choice((True, False, None)) for i in range(3)}
                      for e, _ in pool for s in range(4)}
            read = lambda e, s: scores[e, s]
            for top in (1, 3):
                reference = resolve_constraints(pool, read, top=top, strategy="exhaustive")
                for strategy in (
                    "certificate", "local_certificate", "refutation_first",
                    "short_circuit", "bundled_first", "shortest_first",
                ):
                    self.assertEqual(resolve_constraints(pool, read, top=top, strategy=strategy)["ranking"],
                                     reference["ranking"])

                def traced(strategy):
                    calls = []

                    def observe(e, s):
                        calls.append((e, s))
                        return read(e, s)

                    return resolve_constraints(pool, observe, top=top, strategy=strategy), calls

                self.assertEqual(traced("local_certificate"), traced("certificate"))

    def test_request_requires_role_and_verbatim_residual(self):
        text = "The candidate conducted trials. They must have verified the result. Optional: a reference."
        role, residual, optional = clauses(text)
        owners = [
            ("a", {role: ("paper",), optional: ("reference",)}),
            ("b", {role: ("paper",), residual: ("report",)}),
            ("c", {role: ("paper",), residual: ("report",)}),
        ]
        evidence = {
            ("a", "paper"): {role: True},
            ("b", "paper"): {role: True}, ("b", "report"): {residual: None},
            ("c", "paper"): {role: True}, ("c", "report"): {residual: True},
        }
        for strategy in ("exhaustive", "shortest_first", "certificate", "refutation_first"):
            result = verify_request(text, role, owners, lambda e, s: evidence[e, s],
                                    top=1, strategy=strategy)
            self.assertEqual(result["ranking"], [("c", ((role, "paper"), (residual, "report")))])
            self.assertEqual(result["packet"].optional, (optional,))

    def test_request_without_a_required_clause_abstains(self):
        result = verify_request("Optional: a reference.", None, [("a", {})],
                                lambda *_: self.fail("Unexpected read"), top=1)
        self.assertEqual(result["ranking"], [])
