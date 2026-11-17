import unittest
from collections import Counter

from experiments.bilingual.analyze_stratified_review import intervals


class StratifiedReviewTest(unittest.TestCase):
    def test_weighted_sample_and_population_bounds_are_distinct(self):
        sizes={'large':90,'small':10}
        counts={'large':Counter(supported=1,missing=1),
                'small':Counter(changed_or_unsupported=1,uncertain=1)}
        result=intervals(counts,sizes)
        self.assertAlmostEqual(result['weighted_sample_interval'][0],0.45)
        self.assertAlmostEqual(result['weighted_sample_interval'][1],0.95)
        self.assertEqual(result['finite_population_worst_case'],[0.01,0.99])

    def test_empty_stratum_is_rejected(self):
        with self.assertRaises(ValueError):
            intervals({'a':Counter()},{'a':10})


if __name__=='__main__':unittest.main()
