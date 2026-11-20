import random
import unittest

from expert_search.exact_prompt_trie import pack_prompts


class PromptTrieTest(unittest.TestCase):
    def test_reconstructs_every_leaf_and_its_causal_prefix(self):
        rng = random.Random(37)
        cases = [[[7, 1, 2], [7, 1], [7, 1, 2], [7, 4]]]
        cases.extend([
            [tuple(rng.randrange(3) for _ in range(rng.randrange(1, 12))) for _ in range(8)]
            for _ in range(160)
        ])
        for prompts in cases:
            for common_only in (False, True):
                tree = pack_prompts(prompts, common_only=common_only)
                for leaf, tokens in zip(tree.leaves, prompts, strict=True):
                    ancestors = [j for j, stop in enumerate(tree.subtree_end) if j <= leaf < stop]
                    self.assertEqual(tuple(tree.tokens[j] for j in ancestors), tuple(tokens))
                    self.assertEqual(tuple(tree.depth[j] for j in ancestors), tuple(range(len(tokens))))

    def test_root_and_branches_have_distinct_work_counts(self):
        prompts = [[1, 2, 3], [1, 2, 4], [1, 5]]
        self.assertEqual(len(pack_prompts(prompts).tokens), 5)
        self.assertEqual(len(pack_prompts(prompts, common_only=True).tokens), 6)
