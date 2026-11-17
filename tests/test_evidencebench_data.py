import unittest
from expert_search.evidencebench_data import aspect_coverage


class EvidenceBenchTest(unittest.TestCase):
    def setUp(self):
        self.row=dict(paper_as_candidate_pool=['a','b','c'],sentence_index2aspects={'0':['x'],'1':['x','y'],'2':[]})

    def test_shared_aspect_is_not_counted_twice_and_unselected_aspect_stays_missing(self):
        self.assertEqual(aspect_coverage(self.row,[0])['aspect_recall'],.5)
        self.assertEqual(aspect_coverage(self.row,[0,1])['aspect_recall'],1.)
        self.assertEqual(aspect_coverage(self.row,[])['aspect_recall'],0.)

    def test_duplicate_or_out_of_range_sentence_is_rejected(self):
        for ids in ([0,0],[-1],[3]):
            with self.assertRaises(ValueError):aspect_coverage(self.row,ids)
