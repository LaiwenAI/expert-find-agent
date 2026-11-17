import unittest
from expert_search.feedback_router import FeedbackRouter


class FeedbackRouterTest(unittest.TestCase):
    def test_missing_and_unobserved_context_keep_incumbent(self):
        router=FeedbackRouter(['lexical','dense'],'dense',min_observations=2)
        router.observe('short','lexical',None)
        router.observe('short','lexical',1)
        self.assertEqual(router.decision('short'),{'arm':'dense','status':'insufficient_feedback','counts':{'lexical':1,'dense':0}})
        self.assertEqual(router.decision('long')['arm'],'dense')

    def test_requires_both_arms_and_preserves_context_isolation(self):
        router=FeedbackRouter(['lexical','dense'],'dense',min_observations=2)
        for value in (.8,.6):router.observe('short','lexical',value)
        for value in (.2,.4):router.observe('short','dense',value)
        self.assertEqual(router.decision('short')['arm'],'lexical')
        self.assertEqual(router.decision('long')['status'],'insufficient_feedback')

    def test_tie_keeps_incumbent_and_invalid_values_fail(self):
        router=FeedbackRouter(['a','b'],'b',min_observations=1)
        router.observe('x','a',.5);router.observe('x','b',.5)
        self.assertEqual(router.decision('x')['arm'],'b')
        for value in (float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError):router.observe('x','a',value)
        with self.assertRaises(ValueError):router.observe('x','unknown',1)

if __name__=='__main__':unittest.main()
