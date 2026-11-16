import unittest
from expert_search.benchmark import metrics
class MetricsTest(unittest.TestCase):
    def test_metrics(self):
        r=metrics(['b','a'],{'a':1})
        self.assertEqual(r['mrr@10'],.5)
        self.assertEqual(r['recall@10'],1)
        self.assertAlmostEqual(r['ndcg@10'],1/1.584962500721156)
        self.assertEqual(metrics([],{'a':1})['ndcg@10'],0)
