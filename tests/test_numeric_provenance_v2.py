import unittest
from expert_search.bilingual.numeric_provenance_v2 import audit_numeric_provenance_v2,possible_spaced_negative_values


class SpacedSignProvenanceTest(unittest.TestCase):
    def test_ambiguous_minus_is_not_asserted_as_support(self):
        result=audit_numeric_provenance_v2('operates at - 40 degrees',['s0'],'operates at -40 degrees')
        self.assertEqual(result['absent_from_parent_v1'],1)
        self.assertEqual(result['ambiguous_parent_spaced_sign_values'],1)
        self.assertEqual(result['still_absent_from_parent_values'],0)

    def test_range_dash_is_also_ambiguous(self):
        result=audit_numeric_provenance_v2('range 100 - 40',['s0'],'-40')
        self.assertEqual(result['ambiguous_parent_spaced_sign_values'],1)
        self.assertIn('-4E+1',possible_spaced_negative_values('range 100 - 40'))

    def test_unknown_values_remain_flagged(self):
        result=audit_numeric_provenance_v2('at - 40 degrees',['s0'],'-50')
        self.assertEqual(result['still_absent_from_parent_values'],1)
        self.assertEqual(result['ambiguous_parent_spaced_sign_values'],0)


if __name__=='__main__':unittest.main()
