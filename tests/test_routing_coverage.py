import unittest
from experiments.retrieval.fixed_evidence_routing import preserve_candidate_coverage


class RoutingCoverageTests(unittest.TestCase):
    def test_preserves_one_per_candidate_and_respects_budget(self):
        hits=[{'id':'a','evidence':[{'id':'a1'},{'id':'a2'}]},
              {'id':'b','evidence':[{'id':'b1'}]}, {'id':'c','evidence':[]}]
        self.assertEqual(preserve_candidate_coverage(hits,['a2','foreign'],2),['a2','b1'])
        with self.assertRaises(ValueError):
            preserve_candidate_coverage(hits,[],1)
