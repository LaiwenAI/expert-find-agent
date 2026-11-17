import unittest
from experiments.bilingual.analyze_all_intent_alignment import group_interval


class GroupIntervalTest(unittest.TestCase):
    def test_cluster_resampling_keeps_unequal_cluster_weights(self):
        rows=[dict(group_id='a',value=0.0),dict(group_id='a',value=0.0),dict(group_id='b',value=1.0)]
        result=group_interval(rows,'value',seed=7,repeats=100)
        self.assertEqual(result['queries'],3)
        self.assertEqual(result['company_groups'],2)
        self.assertAlmostEqual(result['mean'],1/3)
        self.assertEqual(result,group_interval(rows,'value',seed=7,repeats=100))


if __name__=='__main__':unittest.main()
