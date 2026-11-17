from pathlib import Path
import unittest
from expert_search.merit_suitability import build_case,gold,parse_final_label


class MeritSuitabilityTest(unittest.TestCase):
    def test_labels_excluded_and_complete_input_preserved(self):
        row=dict(score='5',paper=dict(title='Target',abstract='Abstract text',introduction='Full introduction'),author=dict(papers=[dict(title='Prior',abstract='Own method evidence')]))
        official=dict(action_user_prompt='{{paper_title}}\n{{paper_abstract}}\n{{paper_introduction}}\n{{candidate_history_list}}')
        case=build_case(row,0,official)
        self.assertNotIn('score',case);self.assertNotIn('label',case)
        self.assertIn('Full introduction',case['input']);self.assertIn('Own method evidence',case['input'])
        self.assertEqual(gold(row),1)
        with self.assertRaises(ValueError):gold(dict(score=None))

    def test_missing_final_label_not_silently_unqualified(self):
        self.assertEqual(parse_final_label('1. Method\n[FINAL_LABEL]\n1'),1)
        with self.assertRaises(ValueError):parse_final_label('1. Method\n2. Context')
        with self.assertRaises(ValueError):parse_final_label('[FINAL_LABEL]\n1\n[FINAL_LABEL]\n0')
