import unittest

from expert_search.numeric_evidence import constraint, evaluate_measurement


class NumericEvidenceTest(unittest.TestCase):
    def test_chinese_strict_boundary_and_converted_duration(self):
        bound = constraint("\u8981\u6c42\u4f4e\u4e8e 100 \u6beb\u79d2\u3002")
        self.assertEqual(evaluate_measurement(bound, "0.100 \u79d2"), ("reject", "0.100 \u79d2"))
        self.assertEqual(evaluate_measurement(bound, "0.099 \u79d2"), ("eligible", "0.099 \u79d2"))

    def test_non_strict_chinese_percentage_and_range(self):
        bound = constraint("\u8981\u6c42\u4e0d\u4f4e\u4e8e 90 \uff05\u3002")
        self.assertEqual(evaluate_measurement(bound, "90 percent"), ("eligible", "90 percent"))
        self.assertEqual(evaluate_measurement(bound, "87 \u81f3 91 \uff05"), ("unknown", None))

    def test_legacy_duration_and_unrelated_units(self):
        bound = constraint("The limit is at most 85 ms.")
        self.assertEqual(evaluate_measurement(bound, "The result was 0.084 s."), ("eligible", "0.084 s"))
        self.assertEqual(evaluate_measurement(bound, "The result was 12 W."), ("unknown", None))
