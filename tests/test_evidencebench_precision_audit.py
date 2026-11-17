import unittest

from experiments.retrieval.evidencebench_precision_audit import metrics


class EvidenceBenchPrecisionAuditTest(unittest.TestCase):
    def test_precision_and_absent_results_aspects(self):
        row={'paper_as_candidate_pool':['s']*10,'sentence_index2aspects':{'0':['a'],'1':['a','b']},
             'results_aspect_list_ids':['b','c']}
        result=metrics(row,list(range(10)))
        self.assertEqual(result['sentence_precision_at_10'],.2)
        self.assertEqual(result['results_aspect_recall'],.5)
        row['results_aspect_list_ids']=None
        self.assertIsNone(metrics(row,list(range(10)))['results_aspect_recall'])
        with self.assertRaises(ValueError):metrics(row,[0])


if __name__=='__main__':unittest.main()
