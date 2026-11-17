import unittest
from expert_search.bilingual.numeric_provenance import audit_numeric_provenance


class NumericProvenanceTest(unittest.TestCase):
    def test_missing_values_and_citation_scope(self):
        parent='Temperature 500 degrees. ' + 'x'*610 + ' Finish 20 minutes.'
        report=audit_numeric_provenance(parent,['s0'],'At 500 degrees for 20 minutes and 40 cycles.')
        self.assertEqual(report['detected_query_values'],3)
        self.assertEqual(report['in_cited_values'],1)
        self.assertEqual(report['elsewhere_in_parent_values'],1)
        self.assertEqual(report['absent_from_parent_values'],1)

    def test_escapes_and_bad_sources(self):
        result=audit_numeric_provenance('\u6e29\u5ea6500\u5ea6',['s0'],'At 500 degrees')
        self.assertEqual(result['absent_from_parent_values'],0)
        for sources in ([],['s0','s0'],['s9']):
            with self.assertRaises(ValueError):audit_numeric_provenance('At 500 degrees',sources,'500')


if __name__=='__main__':unittest.main()
