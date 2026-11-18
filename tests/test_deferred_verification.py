import itertools
import unittest

from expert_search.deferred_verification import defer_and_rank


class DeferredVerificationTest(unittest.TestCase):
    def test_dominated_ambiguous_expert_needs_no_recheck(self):
        scores = {("a", "first"): 1., ("b", "other"): 0.}

        def recheck(_):
            self.fail("The lower-ranked source cannot change the output")

        result = defer_and_rank(
            [("a", ["first"]), ("b", ["other"])],
            lambda jobs: [scores[pair] for pair in jobs],
            recheck, support=1, top=1, width=2, band=.25,
        )
        self.assertEqual(result["selected"], [["a", ["first"]]])

    def test_ambiguous_earlier_citation_needs_recheck(self):
        sources = [("a", ["first", "second"])]
        rechecked = []

        def recheck(jobs):
            rechecked.extend(jobs)
            return [1.]

        result = defer_and_rank(
            sources, lambda jobs: [0. if source == "first" else 1. for _, source in jobs],
            recheck, support=1, top=1, width=2, band=.25,
        )
        self.assertEqual(result["selected"], [["a", ["first"]]])
        self.assertEqual(rechecked, [("a", "first")])

    def test_matches_full_reference_for_all_small_score_patterns(self):
        ordered = [(0, ["a", "b"]), (1, ["c", "d"]), (2, ["e"])]
        keys = [(expert, source) for expert, sources in ordered for source in sources]
        for signs in itertools.product((-1., 1.), repeat=len(keys)):
            fixed = dict(zip(keys, signs, strict=True))
            for mask in itertools.product((False, True), repeat=len(keys)):
                dynamic = {key: 0. if ambig else value for key, value, ambig in zip(keys, signs, mask, strict=True)}
                for support in (1, 2):
                    expected = []
                    for expert, sources in ordered:
                        positive = [source for source in sources if fixed[expert, source] > 0]
                        expected.append([expert, positive[:support] if len(positive) >= support else []])
                    expected.sort(key=lambda row: not row[1])
                    result = defer_and_rank(
                        ordered, lambda jobs: [dynamic[key] for key in jobs],
                        lambda jobs: [fixed[key] for key in jobs],
                        support=support, top=2, width=2, band=.25,
                    )
                    self.assertEqual(result["selected"], expected[:2])
