import unittest
import numpy as np
from expert_search.cof_late_interaction import aggregate, query_units


class LateInteractionTest(unittest.TestCase):
    def test_sources_must_be_grouped_within_one_expert(self):
        matrix = np.array([[.9,.1],[.1,.9],[.4,.4]])
        first = aggregate(matrix,[0,1])
        self.assertAlmostEqual(first['expert_min'], .9)
        self.assertAlmostEqual(first['single_source_min'], .1)
        self.assertAlmostEqual(aggregate(matrix,[2])['expert_mean'], .4)

    def test_duplicate_units_and_invalid_memberships(self):
        self.assertEqual(query_units('Same title','Same title'), ['Same title'])
        self.assertEqual(query_units('', 'short abstract'), ['short abstract'])
        self.assertEqual(len(query_units('title', 'a full first sentence with context. another full second sentence with context.')), 3)
        with self.assertRaises(ValueError):
            aggregate(np.array([[1.]]), [])
