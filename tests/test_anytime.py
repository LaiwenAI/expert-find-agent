import random
import unittest
from expert_search.anytime import select

class AnytimeTest(unittest.TestCase):
    def test_matches_exhaustive(self):
        rng=random.Random(42)
        for _ in range(100):
            ids=[str(i) for i in range(12)]
            matrix={i:[rng.random() for _ in range(5)] for i in ids}
            out=select(ids,[1]*5,lambda i,j:matrix[i][j],3)
            expected=sorted(ids,key=lambda i:(-sum(matrix[i]),i))[:3]
            self.assertTrue(out['certified'])
            self.assertEqual(set(out['ranking']),set(expected))
    def test_budget_not_certificate(self):
        out=select(['a','b'],[1,1],lambda i,j:0,1,budget=0)
        self.assertFalse(out['certified'])
    def test_invalid_score(self):
        with self.assertRaises(ValueError):
            select(['a','b'],[1],lambda i,j:float('nan'),1)
