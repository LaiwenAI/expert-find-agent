import unittest
from expert_search.statistics import clustered_paired_interval


class ClusteredStatisticsTest(unittest.TestCase):
    def test_queries_from_same_paper_are_resampled_together(self):
        result=clustered_paired_interval([1.,-1.,1.,-1.], ['a','a','b','b'])
        self.assertEqual(result['percentile_95'], [0.,0.])
        self.assertEqual(result['clusters'], 2)

    def test_original_mean_remains_query_weighted(self):
        result=clustered_paired_interval([1.,1.,0.], ['a','a','b'])
        self.assertAlmostEqual(result['mean'],2/3)
        with self.assertRaises(ValueError):clustered_paired_interval([1.],[])
