import unittest
import numpy as np
from expert_search.owned_vector_index import OwnedVectorIndex, ExactOwnedVectorIndex


class Index:
    ntotal=3
    def search(self,q,k):return np.zeros((1,k)),np.arange(k).reshape(1,-1)


class OwnedVectorIndexTest(unittest.TestCase):
    def setUp(self):
        self.index=OwnedVectorIndex([[1,0],[0,1],[.5,.5]],['s1','s2','s3'],[['a','b'],['c'],['a']],index=Index())

    def test_shared_paper_provenance_and_eligibility_are_preserved(self):
        result=self.index.search_vector([1,0],eligible=['b','c','missing'])
        self.assertEqual([h['id'] for h in result['hits']],['b','c'])
        self.assertEqual(result['hits'][0]['evidence'][0]['id'],'s1')
        self.assertEqual(result['unavailable_experts'],['missing'])

    def test_no_sources_remains_unknown_and_mutable_input_cannot_change_index(self):
        self.assertEqual(self.index.search_vector([1,0],eligible=['missing'])['branch'],'no_sources')
        with self.assertRaises(ValueError):self.index.vectors[0,0]=0

    def test_duplicate_source_ids_and_nonfinite_vectors_rejected(self):
        with self.assertRaises(ValueError):OwnedVectorIndex([[1],[2]],['s','s'],[['a'],['a']],index=Index())
        with self.assertRaises(ValueError):self.index.search_vector([float('nan'),0])

    def test_full_reference_preserves_shared_ownership_and_canonical_ties(self):
        exact=ExactOwnedVectorIndex([[1,0],[1,0],[0,1]],['z','a','b'],[['second','first'],['first'],['third']])
        result=exact.search_vector([1,0])
        self.assertEqual([h['id'] for h in result['hits']],['first','second','third'])
        self.assertEqual([s['id'] for s in result['hits'][0]['evidence']],['a','z'])
        filtered=exact.search_vector([1,0],eligible=['second','missing'])
        self.assertEqual([h['id'] for h in filtered['hits']],['second'])
        self.assertEqual(filtered['unavailable_experts'],['missing'])
