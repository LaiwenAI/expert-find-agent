import unittest
from expert_search.constraint_metrics import evidence_gain,deadline_gain
class ConstraintMetricsTest(unittest.TestCase):
    def test_mandatory_cannot_be_compensated(self):
        c={'a':{'weight':1,'mandatory':True},'b':{'weight':9,'mandatory':False}}
        r=evidence_gain(c,{'b':'supported'})
        self.assertEqual(r['supported_coverage'],.9);self.assertEqual(r['gated_gain'],0)
        self.assertEqual(evidence_gain(c,{'a':'supported','b':'supported'})['gated_gain'],1)
    def test_repeated_judgment_not_double_counted(self):
        self.assertEqual(evidence_gain({'a':{'weight':1}}, {'a':'supported'})['supported_coverage'],1)
    def test_deadline_and_regression(self):
        events=[(1,.8),(3,.2)]
        self.assertEqual(deadline_gain(events,.5),0)
        self.assertEqual(deadline_gain(events,2),.8)
        self.assertEqual(deadline_gain(events,4),.2)
