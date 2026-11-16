import unittest
from expert_search.packing_analysis import paired_interval, distribution


class PackingAnalysisTest(unittest.TestCase):
    def test_constant_paired_effect_and_repeatability(self):
        result = paired_interval([.2] * 5)
        self.assertEqual(result, paired_interval([.2] * 5))
        self.assertAlmostEqual(result['mean'], .2)
        for endpoint in result['percentile_95']:
            self.assertAlmostEqual(endpoint, .2)

    def test_invalid_measurements_not_silently_dropped(self):
        for bad in ([], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                distribution(bad)
        self.assertEqual(distribution([1, 1, 10])['maximum'], 10.)


if __name__ == '__main__':
    unittest.main()
