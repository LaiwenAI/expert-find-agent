import unittest
from experiments.analyze_prioritized_fresh_review import overlay, bounds


class PrioritizedOverlayTest(unittest.TestCase):
    def test_partial_overlay_narrows_bounds_without_mutation(self):
        grades = {'q': {'a': None, 'b': 1}}
        rankings = {'q': {'qwen-paper-max': ['a', 'b'], 'bm25-max': ['b', 'a']}}
        mapping = [{'case_id': 'c', 'query_id': 'q', 'expert_id': 'a'}]
        updated, count = overlay(grades, mapping, {'c': {'annotation': {'grade': 2}}})
        self.assertEqual(count, 1); self.assertIsNone(grades['q']['a'])
        before, after = bounds(rankings, grades), bounds(rankings, updated)
        self.assertGreater(before['upper'] - before['lower'], 0)
        self.assertAlmostEqual(after['lower'], after['upper'])
        self.assertGreaterEqual(after['lower'], before['lower'])
        self.assertLessEqual(after['upper'], before['upper'])
        with self.assertRaises(ValueError):
            overlay({'q': {'a': 1}}, mapping, {'c': {'annotation': {'grade': 2}}})


if __name__ == '__main__':
    unittest.main()
