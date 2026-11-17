import unittest
from expert_search.evaluation_gain import snapshot_gain_interval,actionable_gain_interval,latest_gain_interval,condensed_ndcg


class EvaluationGainTest(unittest.TestCase):
    def test_unjudged_is_not_a_negative_and_no_output_is_not_uncertainty_credit(self):
        unknown=snapshot_gain_interval(['a','b'],{'a':None,'b':None},2)
        self.assertEqual((unknown['lower'],unknown['upper']),(0.,1.))
        zero=snapshot_gain_interval(['a'],{'a':0},1)
        self.assertEqual((zero['lower'],zero['upper']),(0.,0.))
        empty=snapshot_gain_interval([],{'a':None},1)
        self.assertEqual((empty['lower'],empty['upper']),(0.,0.))

    def test_latest_snapshot_regression_and_undefined_ndcg_are_visible(self):
        self.assertEqual(latest_gain_interval([(1,.8,1.),(2,.2,.3)],2),dict(lower=.2,upper=.3))
        self.assertIsNone(condensed_ndcg(['a'],{'a':None}))
        self.assertIsNone(condensed_ndcg(['a'],{'a':0}))
        self.assertEqual(condensed_ndcg(['u','a'],{'u':None,'a':3}),1.)
        with self.assertRaises(ValueError):snapshot_gain_interval(['a','a'],{'a':3})

    def test_topic_adjacency_is_not_actionable_method_credit(self):
        self.assertEqual(actionable_gain_interval(['a'],{'a':1},1)['upper'],0.)
        self.assertEqual(actionable_gain_interval(['a'],{'a':2},1)['lower'],1.)
        self.assertEqual(actionable_gain_interval(['a'],{'a':None},1)['upper'],1.)
