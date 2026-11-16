import unittest
from expert_search.cmu_gold import pairwise_loss
class GoldTest(unittest.TestCase):
    def test_weighted_pairwise_and_missing(self):
        labels={'r':{'a':5,'b':1}}
        self.assertEqual(pairwise_loss({'r':{'a':1,'b':0}},labels)['weighted_pairwise_loss'],0)
        self.assertEqual(pairwise_loss({'r':{'a':0,'b':0}},labels)['weighted_pairwise_loss'],.5)
        self.assertEqual(pairwise_loss({'r':{'a':0,'b':1}},labels)['weighted_pairwise_loss'],1)
        with self.assertRaises(ValueError):pairwise_loss({'r':{'a':1}},labels)
