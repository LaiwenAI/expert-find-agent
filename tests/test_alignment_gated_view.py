import unittest
from experiments.bilingual.export_alignment_gated_view import apply_alignment


class AlignmentGateTest(unittest.TestCase):
    def test_disagreement_downgrades_and_unjudged_not_promoted(self):
        row=dict(id='q',deidentification='same_model_screened_candidate',status='retained_model_screened_pair')
        audit=dict(id='q',arms={'old':{'status':'valid','label':'equivalent'},'new':{'status':'valid','label':'different'}})
        self.assertEqual(apply_alignment(row,audit)['status'],'unjudged_pair')
        audit['arms']['new']['label']='equivalent';row['status']='unjudged_pair'
        self.assertEqual(apply_alignment(row,audit)['status'],'unjudged_pair')
        audit['arms']['new']['status']='unresolved';row['status']='retained_model_screened_pair'
        self.assertEqual(apply_alignment(row,audit)['status'],'unjudged_pair')


if __name__=='__main__':unittest.main()
