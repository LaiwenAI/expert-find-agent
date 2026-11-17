import unittest

from experiments.bilingual.audit_release_identifiers import PATTERNS


class ReleaseIdentifierAuditTest(unittest.TestCase):
    def test_common_literal_indicators_and_nonmatches(self):
        self.assertTrue(PATTERNS['email'].search('contact researcher@example.org'))
        self.assertTrue(PATTERNS['url'].search('see https://example.org/project'))
        self.assertTrue(PATTERNS['long_digit_run'].search('reference 12345678'))
        self.assertTrue(PATTERNS['chinese_organization_suffix'].search('\u67d0\u5927\u5b66'))
        self.assertFalse(PATTERNS['long_digit_run'].search('temperature 1500 K'))


if __name__=='__main__':unittest.main()
