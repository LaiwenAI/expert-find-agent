from itertools import product
import unittest
from expert_search.evaluation_gain import actionable_gain_interval,paired_actionable_gain_interval


class PairedGainTest(unittest.TestCase):
    def test_shared_unknowns_cancel_in_identical_rankings(self):
        r=paired_actionable_gain_interval(['a','b'],['a','b'],{'a':None,'b':None},k=3)
        self.assertEqual((r['lower'],r['upper']),(0.,0.))
        self.assertEqual(r['cancelled_shared_unjudged'],2)
        self.assertLess(r['independent_bounds'][0],0)

    def test_bounds_equal_all_shared_label_completion_extrema(self):
        first=['a','b','d'];second=['b','c','a'];labels={'a':None,'b':None,'c':2,'d':None};r=paired_actionable_gain_interval(first,second,labels)
        outcomes=[]
        for a,b,d in product([0,2],repeat=3):
            complete=dict(labels,a=a,b=b,d=d)
            outcomes.append(actionable_gain_interval(first,complete,3)['lower']-actionable_gain_interval(second,complete,3)['lower'])
        self.assertAlmostEqual(r['lower'],min(outcomes));self.assertAlmostEqual(r['upper'],max(outcomes))
        self.assertGreaterEqual(r['lower'],r['independent_bounds'][0]);self.assertLessEqual(r['upper'],r['independent_bounds'][1])
