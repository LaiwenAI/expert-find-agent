import unittest
from expert_search.bilingual.postvalidation import check_pair


class ComparatorPostvalidationTest(unittest.TestCase):
    def test_negated_comparators_do_not_match_opposite_suffix(self):
        for phrase in ['not below', 'not under', 'not lower than', 'no less than']:
            self.assertEqual(check_pair('至少5 mm', phrase + ' 5 mm')['status'], 'match')
            self.assertEqual(check_pair('低于5 mm', phrase + ' 5 mm')['status'], 'mismatch')
        for phrase in ['not above', 'not over', 'no greater than', 'not higher than']:
            self.assertEqual(check_pair('不超过5 mm', phrase + ' 5 mm')['status'], 'match')
            self.assertEqual(check_pair('高于5 mm', phrase + ' 5 mm')['status'], 'mismatch')

    def test_unsupported_scoped_negation_abstains(self):
        for text in ['not necessarily below 5 mm', 'not not below 5 mm', 'never below 5 mm']:
            self.assertEqual(check_pair('至少5 mm', text)['status'], 'uncertain')
        self.assertEqual(check_pair('至少5 mm', 'not suitable; at least 5 mm')['status'], 'match')

    def test_case_fullwidth_and_numeric_changes(self):
        self.assertEqual(check_pair('不超过１２．５毫米', 'no greater than 12.5 millimeters')['status'], 'match')
        self.assertEqual(check_pair('至少0.5 mm', 'at least 0.5 mM')['status'], 'mismatch')
        self.assertEqual(check_pair('至少0.5 mm', 'at least 5 mm')['status'], 'mismatch')
