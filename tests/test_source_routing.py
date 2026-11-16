import unittest
import numpy as np
from expert_search.source_routing import expand_sources


class Index:
    def search(self,q,k):return np.zeros((1,k)),np.arange(k).reshape(1,-1)


class SourceRoutingTest(unittest.TestCase):
    def test_ineligible_owners_do_not_consume_expert_budget(self):
        owners=[['outside'],['outside'],['a'],['b']]
        result=expand_sources(Index(),[1,0],owners,{'a','b'},target=2,initial=1,limit=4)
        self.assertEqual(result['depths'],[1,2,4])
        self.assertEqual(result['candidates'],['a','b'])
        self.assertTrue(result['target_met'])

    def test_shared_source_recovers_all_eligible_owners_and_limit_remains_explicit(self):
        result=expand_sources(Index(),[1], [['a','b'],['outside']], {'a','b','c'},target=3,initial=1,limit=2)
        self.assertEqual(result['candidates'],['a','b'])
        self.assertFalse(result['target_met'])
