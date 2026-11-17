import unittest
from expert_search.bilingual.constraints import lossless_segments,check_numeric_alignment,numeric_anchors


class BilingualConstraintTest(unittest.TestCase):
    def test_segments_cover_every_original_character_with_real_offsets(self):
        text='标题\n要求不低于0.5 mm。\n保留所有内容；以及尾部。'
        parts=lossless_segments(text,12)
        self.assertEqual(''.join(r['zh'] for r in parts),text)
        for r in parts:self.assertEqual(text[r['start']:r['end']],r['zh'])

    def test_comparator_decimal_and_unit_case_not_erased(self):
        self.assertTrue(check_numeric_alignment('至少0.5 mm','at least 0.5 mm')['exact_ordered_supported_anchors_match'])
        for en in ['at most 0.5 mm','at least 5 mm','at least 0.5 mM']:
            self.assertFalse(check_numeric_alignment('至少0.5 mm',en)['exact_ordered_supported_anchors_match'])
        self.assertEqual(numeric_anchors('阈值１２．５毫米')[0]['literal'],'１２．５')
        self.assertTrue(check_numeric_alignment('不超过１２．５毫米','at most 12.5 millimeters')['exact_ordered_supported_anchors_match'])
