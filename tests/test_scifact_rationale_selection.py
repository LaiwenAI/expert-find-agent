import unittest
from experiments.scifact.rationale_selection_pilot import lexical_order,metrics


class SciFactRationaleTest(unittest.TestCase):
    def test_complete_set_and_fixed_precision_denominator(self):
        groups=[{'sentences':[0,2]},{'sentences':[3]}]
        self.assertEqual(metrics([0,1],groups),{'complete_rationale':0,'annotated_sentence_p_at_k':1/3,'returned':2})
        self.assertEqual(metrics([0,2,1],groups),{'complete_rationale':1,'annotated_sentence_p_at_k':2/3,'returned':3})
        self.assertEqual(metrics([3],groups)['complete_rationale'],1)

    def test_lexical_order_is_deterministic(self):
        self.assertEqual(lexical_order('robot', ['irrelevant','robot robot','robot']),[1,2,0])
        self.assertEqual(lexical_order('absent',['first','second']),[0,1])


if __name__=='__main__':unittest.main()
