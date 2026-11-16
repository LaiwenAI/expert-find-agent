import unittest
from expert_search.qwen_reranker import pack_tokens


class RerankerPackingTest(unittest.TestCase):
    def test_context_truncation_keeps_prefix_and_answer_suffix(self):
        self.assertEqual(pack_tokens([1,2],[3,4,5,6],[7,8],6),([1,2,3,4,7,8],8))
        self.assertEqual(pack_tokens([1],[2],[3],10),([1,2,3],3))
        for length in (0,2,True,2.5):
            with self.assertRaises(ValueError):pack_tokens([1],[2],[3],length)
