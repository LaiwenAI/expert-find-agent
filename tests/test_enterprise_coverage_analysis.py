import hashlib
import unittest

from experiments.retrieval.analyze_enterprise_coverage import frozen_pool_hash, unique_index


class EnterpriseCoverageAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.pools={'q': {'candidates': ['e']}}
        self.snapshots={('q','e'): {'sources': [{'source_id':'s','start':0,'end':4,
                          'sha256':hashlib.sha256(b'text').hexdigest()}]}}
        self.corpus={'s': {'id':'s','expert_id':'e','text':'text'}}

    def test_reconstructed_pool_detects_source_and_owner_changes(self):
        digest,candidates=frozen_pool_hash('q',self.pools,self.snapshots,self.corpus)
        self.assertEqual(len(digest),64)
        self.assertEqual(candidates,{'e'})
        self.corpus['s']['text']='next'
        with self.assertRaises(ValueError):
            frozen_pool_hash('q',self.pools,self.snapshots,self.corpus)
        self.corpus['s'].update(text='text',expert_id='other')
        with self.assertRaises(ValueError):
            frozen_pool_hash('q',self.pools,self.snapshots,self.corpus)

    def test_rejects_duplicate_identities_and_out_of_bounds_spans(self):
        with self.assertRaises(ValueError):
            unique_index([{'id':'s'},{'id':'s'}],lambda row:row['id'])
        self.snapshots['q','e']['sources'][0]['end']=5
        with self.assertRaises(ValueError):
            frozen_pool_hash('q',self.pools,self.snapshots,self.corpus)


if __name__=='__main__':
    unittest.main()
