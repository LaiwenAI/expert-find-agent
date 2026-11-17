import unittest
from expert_search.feedback_router import FeedbackRouter
from expert_search.feedback_events import FeedbackEvents


class FeedbackEventsTest(unittest.TestCase):
    def setUp(self):
        self.router=FeedbackRouter(['lexical','dense'],'dense',min_observations=1)
        self.events=FeedbackEvents(self.router)

    def test_delayed_and_duplicate_feedback_is_counted_once(self):
        self.assertTrue(self.events.expose('one','long','lexical',.5))
        self.assertFalse(self.events.expose('one','long','lexical',.5))
        self.assertFalse(self.events.outcome('one',None))
        self.assertTrue(self.events.outcome('one',.7))
        self.assertFalse(self.events.outcome('one',.7))
        self.assertEqual(self.router.decision('long')['counts'],{'lexical':1,'dense':0})
        self.assertEqual(self.events.snapshot()[0]['outcome'],.7)
        with self.assertRaises(ValueError):self.events.outcome('one',.4)
        with self.assertRaises(ValueError):self.events.expose('one','short','lexical',.5)

    def test_unknown_and_invalid_feedback_does_not_mutate(self):
        with self.assertRaises(ValueError):self.events.outcome('missing',1)
        for propensity in (0,-1,1.1,float('nan'),True):
            with self.assertRaises(ValueError):self.events.expose('bad','x','dense',propensity)
        self.events.expose('pending','x','dense',1)
        with self.assertRaises(ValueError):self.events.outcome('pending',float('inf'))
        self.assertIsNone(self.events.snapshot()[0]['outcome'])
        self.assertEqual(self.router.decision('x')['counts']['dense'],0)

if __name__=='__main__':unittest.main()
