import unittest
from expert_search.shared_evidence import factor_candidates,expand_candidates


class SharedTest(unittest.TestCase):
    def test_roundtrip_preserves_ownership_ids_and_offsets(self):
        candidates=[dict(candidate_id=0,evidence=[dict(paper_id='a',start=2,end=6,text='same')]),
                    dict(candidate_id=1,evidence=[dict(paper_id='b',start=8,end=12,text='same')])]
        result=factor_candidates(candidates)
        self.assertEqual(len(result['source_windows']),1)
        self.assertEqual(expand_candidates(result),candidates)
        result['candidates'][0]['evidence'][0]['source_id']=True
        with self.assertRaises(ValueError):expand_candidates(result)
