import unittest
from expert_search.source_fusion import reciprocal_scores,source_fusion


class SourceFusionTest(unittest.TestCase):
    def test_tied_and_zero_lexical_scores_do_not_gain_arbitrary_id_order(self):
        scores=reciprocal_scores({'z':0.,'a':0.,'hit':2.},positive_only=True)
        self.assertEqual(scores,{'hit':1/61})
        scores=reciprocal_scores({'z':1.,'a':1.})
        self.assertEqual(scores['z'],scores['a'])

    def test_evidence_is_aggregated_only_to_its_owners_without_duplicate_votes(self):
        result=source_fusion({'s1':1.,'s2':.1},{'s2':3.},{'a':['s1','s1'],'b':['s2']})
        self.assertEqual(result['a']['witnesses'],['s1'])
        self.assertEqual(result['a']['source_top3'],result['a']['source_max']/3)
        self.assertGreater(result['b']['source_max'],result['a']['source_max'])
