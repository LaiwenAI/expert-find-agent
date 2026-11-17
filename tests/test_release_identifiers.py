import unittest
from experiments.bilingual.audit_release_identifiers import PATTERNS


class ReleaseIdentifiersTest(unittest.TestCase):
    def test_han_detection_including_supplementary_plane(self):
        pattern=PATTERNS['han_or_radical']
        self.assertIsNotNone(pattern.search('prefix \\u4e2d suffix'.encode().decode('unicode_escape')))
        self.assertIsNotNone(pattern.search(chr(0x20000)))
        self.assertIsNone(pattern.search('English request, 20 kg.'))


if __name__=='__main__':unittest.main()
