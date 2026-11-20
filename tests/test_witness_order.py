import itertools
import random
import unittest

from expert_search.witness_order import stable_witnesses


def emit(scores, owners, cutoff, top):
    exact = {}
    for source in sorted(scores, key=lambda source: (-scores[source], source)):
        if scores[source] > cutoff:
            for owner in owners[source]:
                exact.setdefault(owner, source)
                if len(exact) == top:
                    return tuple(exact.items())
    return tuple(exact.items())


class StableWitnessTest(unittest.TestCase):
    def test_retires_source_that_can_only_repeat_an_already_emitted_owner(self):
        scores = {0: 5, 1: 3, 2: 3}
        owners = {0: ("a",), 1: ("a",), 2: ("b",)}
        self.assertTrue(stable_witnesses(scores, owners, radius=.5, cutoff=0, top=2))

    def test_certificate_matches_exact_corner_replay(self):
        rng = random.Random(42)
        for _ in range(1500):
            scores = {i: rng.randint(-2, 4) for i in range(4)}
            owners = {i: tuple(rng.sample(range(4), rng.randrange(3))) for i in scores}
            radius = rng.choice((0, .5, 1))
            top = rng.choice((1, 2, 4))
            outputs = {
                emit(dict(zip(scores, corners, strict=True)), owners, 0, top)
                for corners in itertools.product(*[(score - radius, score + radius) for score in scores.values()])
            }
            self.assertEqual(
                stable_witnesses(scores, owners, radius=radius, cutoff=0, top=top),
                len(outputs) == 1, (scores, owners, radius, top),
            )
