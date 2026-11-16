import math
import unittest
from expert_search.profile_enrichment import log_tails,EnrichmentIndex


class EnrichmentTest(unittest.TestCase):
    def test_exact_small_hypergeometric_tail(self):
        values=log_tails(5,2,2)
        self.assertEqual(values[0],0)
        self.assertAlmostEqual(math.exp(values[1]),.7)
        self.assertAlmostEqual(math.exp(values[2]),.1)
        self.assertGreaterEqual(values[1],values[2])

    def test_source_membership_and_size_normalization(self):
        index=EnrichmentIndex(list('abcdefghij'),{'small':{'a'},'large':set('abcde')},depths=(1,))
        result=index.score(list('abcdefghij'))
        self.assertGreater(result['scores']['small'],result['scores']['large'])
        self.assertEqual(result['witness']['small']['hits'],1)
        with self.assertRaises(ValueError):index.score(['a','a'])
