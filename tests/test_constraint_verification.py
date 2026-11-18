import random
import unittest

from expert_search.constraint_verification import resolve_constraints


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
                for strategy in ("certificate", "local_certificate", "short_circuit", "bundled_first"):
                    self.assertEqual(resolve_constraints(pool, read, top=top, strategy=strategy)["ranking"],
                                     reference["ranking"])

                def traced(strategy):
                    calls = []

                    def observe(e, s):
                        calls.append((e, s))
                        return read(e, s)

                    return resolve_constraints(pool, observe, top=top, strategy=strategy), calls

                self.assertEqual(traced("local_certificate"), traced("certificate"))
