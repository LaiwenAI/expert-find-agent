import unittest
from expert_search.cof_data import unique_records
from expert_search.cof_benchmark import rank_judged,precision_metrics


class CoFDataTest(unittest.TestCase):
    def test_unjudged_reviewers_never_become_negatives(self):
        labels={'a':1,'b':0};scores={'a':.7,'b':.4,'unjudged':100.}
        self.assertEqual(rank_judged(scores,labels),['a','b'])
        with self.assertRaisesRegex(ValueError,'unjudged'):
            precision_metrics(['unjudged','a'],labels)

    def test_exact_duplicates_are_distinct_from_conflicting_ids(self):
        row={'paper':'a','abstract':'same source text'}
        self.assertEqual(unique_records([row,dict(row)],'paper'),{'a':row})
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            unique_records([row,{'paper':'a','abstract':'different text'}],'paper')
