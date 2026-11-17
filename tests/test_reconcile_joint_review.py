import unittest
from experiments.retrieval.reconcile_joint_review import reconcile


def row(reviewer,method=None,constraint=None,role=None):
    return {'reviewer_id':reviewer,'method_fit':method,'constraint_support':constraint,'personal_contribution':role}


class ReconcileJointTest(unittest.TestCase):
    def test_agreement_missing_uncertainty_and_disagreement(self):
        a={'1':row('one','supported',None,'uncertain'),'2':row('one','supported')}
        b={'1':row('two','supported','supported','uncertain'),'2':row('two','not_supported_by_view')}
        result,counts=reconcile(a,b)
        self.assertEqual(result['1']['method_fit']['status'],'agreed')
        self.assertEqual(result['1']['constraint_support']['status'],'missing_review')
        self.assertEqual(result['1']['personal_contribution']['status'],'agreed_uncertain')
        self.assertEqual(result['2']['method_fit']['status'],'disagreement')
        self.assertIsNone(result['2']['method_fit']['label'])
        self.assertEqual(counts['method_fit']['agreed'],1)

    def test_adjudication_needs_disagreement_and_distinct_reviewer(self):
        a={'1':row('one','supported')};b={'1':row('two','not_supported_by_view')}
        result,_=reconcile(a,b,{'1':row('three','uncertain')})
        self.assertEqual(result['1']['method_fit']['status'],'adjudicated_uncertain')
        with self.assertRaises(ValueError):reconcile(a,b,{'1':row('one','supported')})
        with self.assertRaises(ValueError):reconcile(a,a)
        with self.assertRaises(ValueError):reconcile(a,b,{'1':row('three',None,'supported')})

if __name__=='__main__':unittest.main()
