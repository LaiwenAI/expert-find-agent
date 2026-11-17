import unittest
from expert_search.bilingual.postvalidation import check_pair
from experiments.bilingual.export_deidentified_candidates import apply_candidate


class DeidentifiedCandidateTest(unittest.TestCase):
    def test_never_promotes_unjudged_and_rejects_changed_checks(self):
        row=dict(id='q',english='A firm needs 5 kg.',status='unjudged_pair')
        candidate=dict(id='q',prior_status='unjudged_pair',english='A company needs 5 kg.',has_han=False,
                       numeric_check=check_pair(row['english'],'A company needs 5 kg.'),audit={'identity':'no','meaning':'yes'},
                       status='model_screened_candidate',output_status='unjudged_pair')
        result=apply_candidate(row,candidate)
        self.assertEqual(result['status'],'unjudged_pair')
        self.assertEqual(result['english'],candidate['english'])
        candidate['numeric_check']['status']='mismatch'
        with self.assertRaises(ValueError):apply_candidate(row,candidate)

    def test_unresolved_retains_original_and_marks_flag(self):
        row=dict(id='q',english='A firm needs water.',status='retained_model_screened_pair')
        candidate=dict(id='q',prior_status=row['status'],english=None,status='unresolved')
        result=apply_candidate(row,candidate)
        self.assertEqual(result['english'],row['english'])
        self.assertEqual(result['deidentification'],'flagged_unresolved')


if __name__=='__main__':unittest.main()
