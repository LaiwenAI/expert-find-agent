import unittest
from expert_search.cof_crossencoder import checked_ranking,source_max_scores


class CrossencoderTest(unittest.TestCase):
    def test_candidate_integrity_and_deterministic_ties(self):
        aliases=['z','a']
        self.assertEqual(checked_ranking([dict(id=0,score=.5),dict(id=1,score=.5)],aliases),['a','z'])
        for results in ([dict(id=0,score=.5)]*2,[dict(id=0,score=float('nan')),dict(id=1,score=.2)]):
            with self.assertRaises(ValueError):checked_ranking(results,aliases)

    def test_shared_source_scores_preserve_candidate_ownership(self):
        payload=dict(source_windows=[dict(source_id=0),dict(source_id=1)],
                     candidates=[dict(candidate_id=0,evidence=[dict(source_id=0)]),
                                 dict(candidate_id=1,evidence=[dict(source_id=0),dict(source_id=1)])])
        self.assertEqual(source_max_scores([dict(id=0,score=.2),dict(id=1,score=.8)],payload),
                         [dict(id=0,score=.2),dict(id=1,score=.8)])
