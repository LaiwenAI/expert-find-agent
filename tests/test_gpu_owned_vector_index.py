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
            with self.assertRaises(ValueError):gpu.search_vector([1,0],eligible=['first'])
            with self.assertRaises(ValueError):gpu.search_vector([np.nan,0])
            torch.backends.cuda.matmul.allow_tf32=True
            with self.assertRaises(ValueError):gpu.search_vector([1,0])
        finally:
            torch.backends.cuda.matmul.allow_tf32=old


if __name__=='__main__':unittest.main()
