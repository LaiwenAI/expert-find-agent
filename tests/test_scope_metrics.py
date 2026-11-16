import unittest
from expert_search.scope_metrics import scoped_gain


class ScopeTest(unittest.TestCase):
    def setUp(self):
        self.plan={k:dict(weight=1,mandatory=True,kind='atomic') for k in ('a','b')}
        self.plan['joint']=dict(weight=1,mandatory=True,kind='relation',dependencies=['a','b'])

    def test_independent_capabilities_do_not_prove_application(self):
        labels={'a':'supported','b':'supported'}
        result=scoped_gain(self.plan,labels)
        self.assertEqual(result['atomic_coverage'],1)
        self.assertEqual(result['relation_coverage'],0)
        self.assertEqual(result['gated_gain'],0)
        self.assertEqual(scoped_gain({k:self.plan[k] for k in ('a','b')},labels)['gated_gain'],1)
        labels['joint']='supported'
        self.assertEqual(scoped_gain(self.plan,labels)['gated_gain'],1)

    def test_missing_endpoint_and_cycles(self):
        result=scoped_gain(self.plan,{'a':'supported','joint':'supported'})
        self.assertEqual(result['resolved_judgments']['joint'],'unknown')
        self.plan['a']=dict(weight=1,mandatory=True,kind='relation',dependencies=['b','joint'])
        with self.assertRaisesRegex(ValueError,'Cyclic'):scoped_gain(self.plan,{})
