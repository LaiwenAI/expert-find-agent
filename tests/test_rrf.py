import unittest
from expert_search.hybrid_benchmark import rrf
class RRFTest(unittest.TestCase):
    def test_overlap_and_ties(self):
        self.assertEqual(rrf([['a','b'],['c','b']],3)[0],'b')
        self.assertEqual(rrf([['b'],['a']],2),['a','b'])
        with self.assertRaises(ValueError):rrf([['a','a']])
