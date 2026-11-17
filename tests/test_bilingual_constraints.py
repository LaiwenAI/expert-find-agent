import unittest
from expert_search.bilingual.constraints import lossless_segments,check_numeric_alignment,numeric_anchors


class BilingualConstraintTest(unittest.TestCase):
    def test_segments_cover_every_original_character_with_real_offsets(self):
        text='\u6807\u9898\n\u8981\u6c42\u4e0d\u4f4e\u4e8e0.5 mm\u3002\n\u4fdd\u7559\u6240\u6709\u5185\u5bb9\uff1b\u4ee5\u53ca\u5c3e\u90e8\u3002'
        parts=lossless_segments(text,12)
        self.assertEqual(''.join(r['zh'] for r in parts),text)
        for r in parts:self.assertEqual(text[r['start']:r['end']],r['zh'])

    def test_comparator_decimal_and_unit_case_not_erased(self):
        self.assertTrue(check_numeric_alignment('\u81f3\u5c110.5 mm','at least 0.5 mm')['exact_ordered_supported_anchors_match'])
        for en in ['at most 0.5 mm','at least 5 mm','at least 0.5 mM']:
            self.assertFalse(check_numeric_alignment('\u81f3\u5c110.5 mm',en)['exact_ordered_supported_anchors_match'])
        self.assertEqual(numeric_anchors('\u9608\u503c\uff11\uff12\uff0e\uff15\u6beb\u7c73')[0]['literal'],'\uff11\uff12\uff0e\uff15')
        self.assertTrue(check_numeric_alignment('\u4e0d\u8d85\u8fc7\uff11\uff12\uff0e\uff15\u6beb\u7c73','at most 12.5 millimeters')['exact_ordered_supported_anchors_match'])
