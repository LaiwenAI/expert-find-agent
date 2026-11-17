import unittest
from expert_search.review_agreement import quadratic_kappa,reconcile


def review(grade,annotator,owner=None):
    return {'annotation':{'grade':grade,'annotator':annotator,'source_owner_verified':owner}}


class ReviewAgreementTest(unittest.TestCase):
    def test_agreement_adjudication_and_missing(self):
        first={'a':review(2,'A',True),'b':review(1,'A',True),'c':review(None,None)}
        second={'a':review(2,'B',False),'b':review(3,'B',True),'c':review(2,'B')}
        third={'a':review(None,None),'b':review(2,'C'),'c':review(None,None)}
        result=reconcile(first,second,third)
        self.assertEqual(result['outcomes'],{'a':{'grade':2,'status':'agreed'},'b':{'grade':2,'status':'adjudicated'},'c':{'grade':None,'status':'missing_review'}})
        self.assertEqual(result['paired_grades'],2);self.assertEqual(result['exact_agreement'],0.5)
        self.assertEqual(result['ownership_disagreements'],1)

    def test_unresolved_conflict_and_duplicate_reviewer(self):
        a={'x':review(0,'A')};b={'x':review(3,'B')}
        self.assertIsNone(reconcile(a,b)['outcomes']['x']['grade'])
        with self.assertRaises(ValueError):reconcile(a,{'x':review(3,'A')})
        with self.assertRaises(ValueError):reconcile(a,b,{'x':review(1,'A')})

    def test_kappa_and_undefined(self):
        self.assertAlmostEqual(quadratic_kappa([(0,0),(3,3)]),1.)
        self.assertAlmostEqual(quadratic_kappa([(0,3),(3,0)]),-1.)
        self.assertIsNone(quadratic_kappa([(2,2),(2,2)]))
        self.assertIsNone(quadratic_kappa([]))


if __name__=='__main__':unittest.main()
