import math
import unittest
from expert_search.cost_metrics import tokens_per_correctness, incomplete_tokens_per_correctness_bounds


class CostMetricsTests(unittest.TestCase):
    def test_counts_all_attempts_and_zero_correct(self):
        rows=[dict(correct=True,output_tokens=100,input_tokens=400,search_calls=1),
              dict(correct=False,output_tokens=50,input_tokens=200,search_calls=0)]
        self.assertEqual(tokens_per_correctness(rows),800)
        rows[0]['correct']=False
        self.assertTrue(math.isinf(tokens_per_correctness(rows)))

    def test_rejects_unjudged(self):
        with self.assertRaises(ValueError):
            tokens_per_correctness([dict(correct=None,output_tokens=0,input_tokens=0,search_calls=0)])

    def test_incomplete_correctness_bounds_include_zero_correct(self):
        rows=[dict(correct=None,output_tokens=10,input_tokens=0,search_calls=0),
              dict(correct=False,output_tokens=10,input_tokens=0,search_calls=0)]
        lower,upper=incomplete_tokens_per_correctness_bounds(rows)
        self.assertEqual(lower,20)
        self.assertTrue(math.isinf(upper))
