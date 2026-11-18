import unittest

import numpy as np

from expert_search.bounded_adjustment import adjusted_ranking


class BoundedAdjustmentTest(unittest.TestCase):
    def test_dominant_candidate_certifies_after_first_block(self):
        checked = []

        def evaluate(start, end):
            checked.append((start, end))
            return np.zeros(end - start)

        ranks, count = adjusted_ranking([1., .6, .2, 0.], evaluate, weight=.1, top=1, block_size=2)
        self.assertEqual((ranks, count, checked), ([0], 2, [(0, 2)]))

    def test_equal_bound_keeps_later_candidate_eligible(self):
        ranks, count = adjusted_ranking([1., .75], lambda start, end: [-200., 200.][start:end],
                                        weight=.125, top=1, block_size=1)
        self.assertEqual((ranks, count), ([0], 2))

    def test_reversal_and_unresolved_suffix_match_exhaustive(self):
        score = [1., .91, .9, .3, .21, 0.]
        margins = [-4., 4., 0., 10., 7., 6.]
        args = dict(weight=.15, top=2, block_size=2)
        bounded = adjusted_ranking(score, lambda a, b: margins[a:b], **args)
        exhaustive = adjusted_ranking(score, lambda a, b: margins[a:b], exhaust=True, **args)
        self.assertEqual(bounded, ([1, 2], 4))
        self.assertEqual(exhaustive, ([1, 2], 6))
