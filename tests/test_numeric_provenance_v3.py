import unittest
from expert_search.bilingual.numeric_provenance_v3 import audit_numeric_provenance_v3


class CrossLanguageHintTest(unittest.TestCase):
    def test_dimension_hint_is_ambiguous(self):
        result=audit_numeric_provenance_v3('\u4e09\u7ef4\u91cd\u5efa',['s0'],'3D reconstruction')
        self.assertEqual(result['possible_cross_language_values'],1)
        self.assertEqual(result['still_absent_after_hints'],0)

    def test_hundred_meters_hint_requires_matching_expression(self):
        parent='\u8d85\u767e\u7c73\u53f6\u7247'
        self.assertEqual(audit_numeric_provenance_v3(parent,['s0'],'over 100 meters')['possible_cross_language_values'],1)
        self.assertEqual(audit_numeric_provenance_v3(parent,['s0'],'100 kg')['still_absent_after_hints'],1)
        self.assertEqual(audit_numeric_provenance_v3('other',['s0'],'100 meters')['still_absent_after_hints'],1)

    def test_unchanged_known_and_unknown_values(self):
        result=audit_numeric_provenance_v3('\u4e09\u7ef4 and 40',['s0'],'3D at 40 and 80')
        self.assertEqual(result['possible_cross_language_values'],1)
        self.assertEqual(result['still_absent_after_hints'],1)


if __name__=='__main__':unittest.main()
