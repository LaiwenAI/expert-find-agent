import unittest
from experiments.retrieval.audit_atomic_claim_coverage import covered_characters


class AtomicClaimCoverageTest(unittest.TestCase):
    def test_overlaps_and_first_occurrence(self):
        result=covered_characters('abc abc!',[{'text':'abc'},{'text':'bc'}])
        self.assertEqual(result['meaningful_characters'],6)
        self.assertEqual(result['covered_characters'],3)
        self.assertEqual(result['ambiguous_claims'],2)

    def test_rejects_invalid_claim(self):
        for text in ('missing','',None):
            with self.assertRaises(ValueError):covered_characters('source',[{'text':text}])


if __name__=='__main__':unittest.main()
