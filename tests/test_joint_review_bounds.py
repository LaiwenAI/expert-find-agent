import unittest
from experiments.retrieval.joint_review_bounds import calculate,joint_label


class JointBoundsTest(unittest.TestCase):
    def test_shared_unknown_cancels(self):
        result=calculate({'q':['a','b']},{'q':['a','c']},{},k=2)
        self.assertEqual((result['lower'],result['upper']),(-.5,.5))
        self.assertEqual(result['unknown_nonzero_cells'],2)

    def test_known_and_unknown_conjunction(self):
        def fields(a,b):return {'method_fit':{'label':a},'constraint_support':{'label':b}}
        self.assertEqual(joint_label(fields('supported','supported')),1)
        self.assertEqual(joint_label(fields('not_supported_by_view',None)),0)
        self.assertIsNone(joint_label(fields('supported','uncertain')))
        with self.assertRaises(ValueError):joint_label(fields('not_supported_by_view','bad'))
        result=calculate({'q':['a','b']},{'q':['a','c']},{('q','b'):1,('q','c'):0},k=2)
        self.assertEqual((result['lower'],result['upper']),(.5,.5))

    def test_invalid_rankings_rejected(self):
        with self.assertRaises(ValueError):calculate({'q':['a','a']},{'q':['a','b']},{},k=2)

if __name__=='__main__':unittest.main()
