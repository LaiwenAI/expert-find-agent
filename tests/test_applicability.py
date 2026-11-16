import unittest
from expert_search.applicability import assess_candidate,rank_candidates


class ApplicabilityTest(unittest.TestCase):
    def setUp(self):
        self.sources={'s':dict(expert_id='e',text='A method for image recognition.')}
        self.citation=dict(evidence_id='s',start=0,end=31,quote=self.sources['s']['text'])

    def test_method_transfer_keeps_gap_without_claiming_past_target_achievement(self):
        result=assess_candidate('e',{'r':dict(weight=1)}, {'r':dict(level='transferable_method',citations=[self.citation],transfer_reason='Image method may transfer to target images.',assumptions=['Target data require validation.'])},self.sources)
        self.assertEqual(result['actionable_coverage'],1)
        self.assertEqual(result['direct_coverage'],0)
        self.assertEqual(result['unresolved_requirements'],['r'])

    def test_generic_similarity_cannot_outrank_evidenced_transfer(self):
        topic=dict(expert_id='a',actionable_coverage=0,direct_coverage=0,topic_coverage=1)
        transfer=dict(expert_id='b',actionable_coverage=.5,direct_coverage=0,topic_coverage=0)
        self.assertEqual(rank_candidates([topic,transfer])[0]['expert_id'],'b')

    def test_wrong_owner_and_unjustified_transfer_rejected(self):
        a=dict(level='transferable_method',citations=[self.citation],transfer_reason='Possible',assumptions=[])
        with self.assertRaises(ValueError):assess_candidate('e',{'r':{}},{'r':a},self.sources)
        with self.assertRaises(ValueError):assess_candidate('other',{'r':{}},{'r':dict(a,assumptions=['Needs testing'])},self.sources)
