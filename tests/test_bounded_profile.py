import unittest
import numpy as np
from expert_search.bounded_profile import BoundedProfile
class BoundedProfileTest(unittest.TestCase):
    def test_exact_under_random_and_identical(self):
        rng=np.random.default_rng(42)
        for x in [rng.normal(size=(50,12)),np.ones((10,12)),rng.normal(size=(1,12))]:
            idx=BoundedProfile(x)
            for _ in range(30):
                q=rng.normal(size=12);r=idx.maximum(q)
                self.assertAlmostEqual(r['score'],float(np.max(x@q)),places=10)
                self.assertEqual(r['member_dot_products']+r['skipped_members'],len(x))

    def test_stable_dot_preserves_score_across_duplicate_source_partitions(self):
        rng=np.random.default_rng(19)
        base=rng.normal(size=(12,1024))
        x=np.concatenate([base,base[[0,4,8]]])
        for q in (base[0],rng.normal(size=1024)):
            expected=float(np.max(np.einsum('ij,j->i',x,q,optimize=False)))
            for groups in (1,4,12):
                self.assertEqual(BoundedProfile(x,groups=groups).maximum(q,stable_dots=True)['score'],expected)
