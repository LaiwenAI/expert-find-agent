import unittest
from expert_search.pooled_atomic_audit import aggregate


class OpenWorldTest(unittest.TestCase):
    def test_missing_and_failed_judgments_not_negative_capabilities(self):
        self.assertEqual(aggregate(['supported','supported']),'supported')
        self.assertEqual(aggregate(['supported','unknown']),'unknown')
        self.assertIsNone(aggregate(['supported',None]))
        self.assertIsNone(aggregate(['unknown',None]))
        with self.assertRaises(ValueError):aggregate(['contradicted'])
