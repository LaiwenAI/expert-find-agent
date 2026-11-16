import unittest
from expert_search.cof_packing_experiment import validate_order


class OrderTest(unittest.TestCase):
    def test_permutation_rejects_partial_duplicate_and_boolean_aliases(self):
        self.assertEqual(validate_order([2,0,1],3),[2,0,1])
        for order in ([0,1],[0,1,1],[False,1,2],None):
            with self.assertRaises(ValueError):validate_order(order,3)
