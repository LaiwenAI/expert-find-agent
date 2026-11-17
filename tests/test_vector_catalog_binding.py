import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from expert_search.expert_catalog import ExpertCatalog
from expert_search.grounded_pipeline import GroundedPipeline
from expert_search.owned_vector_index import ExactOwnedVectorIndex,OwnedVectorIndex
from expert_search.vector_catalog_binding import VectorCatalogBinding


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class BindingTest(unittest.TestCase):
    def make(self,root):
        corpus=root/'corpus';corpus.mkdir();docs=[dict(id='s1',expert_id='a',text='Alpha method.'),dict(id='s2',expert_id='b',text='Beta method.')]
        (corpus/'evidence.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in docs));catalog=ExpertCatalog(corpus,root/'catalog')
        branches=root/'branches.jsonl';branches.write_text(''.join(json.dumps(dict(id=d['id'],expert_id=d['expert_id'],text=d['text'],source_prefix_sha256=hashlib.sha256(d['text'].encode()).hexdigest()))+'\n' for d in docs))
        vectors=root/'vectors.npy';array=np.array([[1,0],[0,1]],dtype=np.float32);np.save(vectors,array)
        index=ExactOwnedVectorIndex(array,['s1','s2'],[['a'],['b']]);return catalog,branches,vectors,index

    def bind(self,catalog,branches,vectors,index):
        return VectorCatalogBinding(index,catalog,branches,vectors,branches_sha256=sha(branches),vectors_sha256=sha(vectors))

    def test_detects_wrong_vectors_and_preexisting_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog,branches,vectors,index=self.make(root);binding=self.bind(catalog,branches,vectors,index)
            self.assertEqual(binding.sources,2);binding.ensure_current(index,catalog)
            from types import SimpleNamespace
            approximate=OwnedVectorIndex(np.array([[1,0],[0,1]],dtype=np.float32),['s1','s2'],[['a'],['b']],index=SimpleNamespace(ntotal=2))
            with self.assertRaisesRegex(ValueError,'exact'):self.bind(catalog,branches,vectors,approximate)
            wrong_owner=ExactOwnedVectorIndex(np.array([[1,0],[0,1]],dtype=np.float32),['s1','s2'],[['b'],['a']])
            with self.assertRaisesRegex(ValueError,'ownership'):self.bind(catalog,branches,vectors,wrong_owner)
            bad=ExactOwnedVectorIndex(np.array([[0,1],[1,0]],dtype=np.float32),['s1','s2'],[['a'],['b']])
            with self.assertRaisesRegex(ValueError,'vectors'):self.bind(catalog,branches,vectors,bad)
            rows=[json.loads(x) for x in branches.read_text().splitlines()];rows[0]['text']='Wrong';rows[0]['source_prefix_sha256']=hashlib.sha256(b'Wrong').hexdigest();branches.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaisesRegex(ValueError,'prefix'):self.bind(catalog,branches,vectors,index)
            with self.assertRaisesRegex(ValueError,'changed'):binding.ensure_current(index,catalog)
            catalog.db.close()

    def test_strict_pipeline_rejects_unbound_and_stale_snapshots(self):
        class Encoder:
            def encode(self,texts,query=False):return np.array([[1,0]],dtype=np.float32)
        class Planner:
            def plan(self,demand):return dict(requirements={'r':dict(text=demand)})
        class Assessor:
            def assess(self,demand,eid,requirements,sources):return dict(status='structurally_valid',result=dict(expert_id=eid,actionable_coverage=0,direct_coverage=0,topic_coverage=1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog,branches,vectors,index=self.make(root);binding=self.bind(catalog,branches,vectors,index)
            with self.assertRaises(ValueError):GroundedPipeline(Encoder(),index,catalog,Planner(),Assessor(),require_binding=True)
            pipeline=GroundedPipeline(Encoder(),index,catalog,Planner(),Assessor(),binding=binding,require_binding=True)
            events=pipeline.run('Find a method',k=1,candidates=1)
            self.assertEqual(next(e for e in events if e['stage']=='retrieval')['index_catalog_binding'],'validated_at_initialization')
            self.assertEqual(events[-1]['status'],'provisional')
            vectors.write_bytes(vectors.read_bytes()+b'x')
            events=pipeline.run('Find a method',k=1,candidates=1)
            self.assertEqual(events[-1]['stage'],'binding');self.assertEqual(events[-1]['status'],'unresolved')
            catalog.db.close()


if __name__=='__main__':unittest.main()
