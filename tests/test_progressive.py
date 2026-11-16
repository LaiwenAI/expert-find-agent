import unittest
import numpy as np
from expert_search.progressive_dense import progressive_search
class ProgressiveTest(unittest.TestCase):
    def test_full_shortlist_exact(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(20,12));x/=np.linalg.norm(x,axis=1,keepdims=True)
        q=rng.normal(size=12);q/=np.linalg.norm(q);ids=[str(i) for i in range(20)]
        expected=[ids[i] for i in sorted(range(20),key=lambda i:(-float(x[i]@q),ids[i]))[:5]]
        self.assertEqual(progressive_search(x,q,ids,5,4,20),expected)
        with self.assertRaises(ValueError):progressive_search(x,q,ids,5,13,20)

class PartialSelectionTest(unittest.TestCase):
    def test_cutoff_ties(self):
        from expert_search.progressive_dense import top_indices,ProgressiveIndex
        scores=np.array([1.,2.,2.,2.,-1.]);ids=['z','c','a','b','d']
        self.assertEqual(top_indices(scores,ids,2),[2,3])
        idx=ProgressiveIndex(np.eye(5,dtype=np.float32),ids,3)
        q=np.array([1,0,0,0,0],dtype=np.float32)
        self.assertEqual(idx.search(q,2,5),idx.full(q,2))
