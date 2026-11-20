import unittest
import numpy as np
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex

try:
    import torch
except ImportError:
    torch=None


class GpuOwnedVectorIndexTest(unittest.TestCase):
    def test_rejects_non_reference_before_gpu_setup(self):
        if torch is None:self.skipTest('Optional torch is unavailable')
        with self.assertRaises(ValueError):GpuOwnedVectorIndex(object())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA is unavailable')
    def test_cuda_output_and_validation(self):
        old=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            reference=ExactOwnedVectorIndex([[1,0],[1,0],[0,1]],['z','a','b'],[['second','first'],['first'],['third']])
            gpu=GpuOwnedVectorIndex(reference)
            for query in ([1,0],[0,1],[.25,.75]):
                cpu_hits=reference.search_vector(query,k=3)['hits'];gpu_hits=gpu.search_vector(query,k=3)['hits']
                self.assertEqual([x['id'] for x in cpu_hits],[x['id'] for x in gpu_hits])
                self.assertEqual([[e['id'] for e in x['evidence']] for x in cpu_hits],[[e['id'] for e in x['evidence']] for x in gpu_hits])
                for a,b in zip(cpu_hits,gpu_hits):self.assertAlmostEqual(a['score'],b['score'],places=6)
            for eligible in (['first'],['first','second','missing'],['third','first','first'],['missing'],[]):
                a=reference.search_vector([1,0],k=3,eligible=eligible)
                b=gpu.search_vector([1,0],k=3,eligible=eligible)
                self.assertEqual([r['id'] for r in a['hits']],[r['id'] for r in b['hits']])
                self.assertEqual(a['unavailable_experts'],b['unavailable_experts'])
                self.assertEqual([[e['id'] for e in r['evidence']] for r in a['hits']],[[e['id'] for e in r['evidence']] for r in b['hits']])
            extended = ExactOwnedVectorIndex(
                [[1, 0], [.9, .1], [.8, .2], [.7, .3], [0, 1]],
                ["a", "b", "c", "d", "e"],
                [["first"], ["first"], ["first"], ["first"], ["second"]],
            )
            extended_gpu = GpuOwnedVectorIndex(extended)
            for eligible in (None, ["first"]):
                default = extended_gpu.search_vector([1, 0], eligible=eligible)["hits"]
                wide = extended_gpu.search_vector([1, 0], eligible=eligible, evidence_limit=4)["hits"]
                self.assertEqual([hit["id"] for hit in default], [hit["id"] for hit in wide])
                self.assertEqual(default[0]["evidence"], wide[0]["evidence"][:3])
                self.assertEqual([row["id"] for row in wide[0]["evidence"]], ["a", "b", "c", "d"])
            with self.assertRaises(ValueError):
                extended_gpu.search_vector([1, 0], evidence_limit=0)
            with self.assertRaises(ValueError):gpu.search_vector([1,0],eligible=[1])
            with self.assertRaises(ValueError):gpu.search_vector([np.nan,0])
            torch.backends.cuda.matmul.allow_tf32=True
            with self.assertRaises(ValueError):gpu.search_vector([1,0])
        finally:
            torch.backends.cuda.matmul.allow_tf32=old


if __name__=='__main__':unittest.main()
