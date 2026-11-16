import unittest
from expert_search.scope_metrics import scoped_gain


class ScopeFidelityTest(unittest.TestCase):
    def test_lost_application_scope_blocks_atom_and_dependent_relation(self):
        requirements={'a':dict(weight=1,mandatory=False,kind='atomic'),
            'b':dict(weight=1,mandatory=False,kind='atomic'),
            'ab':dict(weight=1,mandatory=False,kind='relation',dependencies=['a','b'])}
        labels={key:'supported' for key in requirements}
        result=scoped_gain(requirements,labels,{'a':'lost','b':'preserved','ab':'preserved'})
        self.assertEqual(result['resolved_judgments'],{'a':'unknown','b':'supported','ab':'unknown'})
        self.assertAlmostEqual(result['supported_coverage'],1/3)
        self.assertEqual(scoped_gain(requirements,labels)['supported_coverage'],1)

    def test_unreviewed_scope_is_unknown_when_scope_audit_requested(self):
        r={'a':dict(weight=1,mandatory=False,kind='atomic')}
        self.assertEqual(scoped_gain(r,{'a':'supported'},{})['supported_coverage'],0)
        with self.assertRaises(ValueError):scoped_gain(r,{'a':'supported'},{'a':'valid'})
