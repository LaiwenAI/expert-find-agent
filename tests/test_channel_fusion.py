import unittest
from expert_search.channel_fusion import fuse


class ChannelFusionTest(unittest.TestCase):
    def test_rrf_retains_channel_ranks_and_rewards_shared_candidate(self):
        result=fuse(dict(paper=['p','shared'],profile=['r','shared']),k=3)
        self.assertEqual([r['id'] for r in result],['shared','p','r'])
        self.assertEqual(result[0]['channel_ranks'],dict(paper=2,profile=2))

    def test_interleaving_deduplicates_without_filling_missing_channel(self):
        result=fuse(dict(paper=['a','b','c'],profile=['b','d']),k=4,method='interleave')
        self.assertEqual([r['id'] for r in result],['a','b','d','c'])
        self.assertEqual(fuse(dict(paper=[],profile=['p']),k=3)[0]['channel_ranks'],dict(profile=1))

    def test_depth_and_invalid_duplicate_inputs(self):
        result=fuse(dict(paper=['a','hidden'],profile=['b']),depth=1,k=10)
        self.assertEqual({r['id'] for r in result},{'a','b'})
        with self.assertRaises(ValueError):fuse(dict(paper=['a','a']))
        with self.assertRaises(ValueError):fuse(dict(paper=['a']),constant=float('nan'))
