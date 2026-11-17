import unittest

from experiments.bilingual.check_validation_containment import analyze


class ValidationContainmentTest(unittest.TestCase):
    def test_embedded_need_and_split_guard(self):
        rows = [dict(id='c', split='construction', group_id='one', zh='prefixabcdefghsuffix'),
                dict(id='v', split='validation', group_id='two', zh='abcdefgh'),
                dict(id='s', split='sealed', group_id='one', zh='unused')]
        records, report = analyze(rows)
        self.assertEqual(records[0]['covered_fraction'], 1.0)
        self.assertEqual(report['threshold_counts']['0.9'], 1)
        rows[1]['group_id'] = 'one'
        with self.assertRaises(ValueError):
            analyze(rows)


if __name__ == '__main__':
    unittest.main()
