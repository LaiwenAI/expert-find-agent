import unittest
import numpy as np
from expert_search.grounded_pipeline import GroundedPipeline


class Encoder:
    def encode(self,texts,query=False):return np.array([[1,0]],dtype=np.float32)


class Index:
    def search_vector(self,q,k):return dict(hits=[dict(id='a',evidence=[dict(id='s')]),dict(id='b',evidence=[dict(id='foreign')])])


class Catalog:
    def for_experts(self,ids):return [dict(id='s',expert_id='a',text='Documented method.'),dict(id='foreign',expert_id='a',text='Owned by another expert.')]


class Planner:
    def plan(self,demand):return dict(requirements={'r':dict(text=demand)})


class Assessor:
    def __init__(self):self.calls=[]
    def assess(self,demand,eid,requirements,sources):
        self.calls.append(eid)
        return dict(status='structurally_valid',result=dict(expert_id=eid,actionable_coverage=0,direct_coverage=0,topic_coverage=1))


class GroundedPipelineTest(unittest.TestCase):
    def test_foreign_witness_cannot_become_expert_evidence_and_failure_is_retained(self):
        assessor=Assessor();events=GroundedPipeline(Encoder(),Index(),Catalog(),Planner(),assessor).run('Find a method.',k=1,candidates=2)
        self.assertEqual(assessor.calls,['a'])
        self.assertEqual(events[-1]['status'],'partial')
        self.assertEqual(events[-1]['unresolved_experts'],['b'])
        self.assertEqual(events[-1]['ranking'][0]['expert_id'],'a')
        self.assertEqual([e['total_seconds'] for e in events],sorted(e['total_seconds'] for e in events))

    def test_targeted_reads_preserve_unresolved_foreign_and_missing_sources(self):
        class TargetCatalog(Catalog):
            def __init__(self):self.requested=None
            def for_evidence(self,ids):
                self.requested=ids
                return self.for_experts([])
        catalog=TargetCatalog();assessor=Assessor()
        events=GroundedPipeline(Encoder(),Index(),catalog,Planner(),assessor).run('Find a method.',k=1,candidates=2)
        self.assertEqual(catalog.requested,['s','foreign'])
        self.assertEqual(events[-1]['unresolved_experts'],['b'])
        self.assertEqual(assessor.calls,['a'])

    def test_catalog_failure_is_explicit_and_stops_assessment(self):
        class ChangedCatalog(Catalog):
            def for_evidence(self,ids):raise ValueError('Snapshot changed')
        assessor=Assessor();events=GroundedPipeline(Encoder(),Index(),ChangedCatalog(),Planner(),assessor).run('Find a method.',k=1,candidates=2)
        self.assertEqual(events[-1]['stage'],'source_reading')
        self.assertEqual(events[-1]['status'],'unresolved')
        self.assertEqual(events[-1]['error_type'],'ValueError')
        self.assertEqual(assessor.calls,[])

    def test_encoding_and_retrieval_errors_are_explicit(self):
        class FailedEncoder:
            def encode(self,texts,query=False):raise TimeoutError()
        class InvalidEncoder:
            def encode(self,texts,query=False):return [[float('nan'),0]]
        class FailedIndex:
            def search_vector(self,q,k):raise RuntimeError()
        class DuplicateIndex:
            def search_vector(self,q,k):return dict(hits=[dict(id='a',evidence=[dict(id='s')]),dict(id='a',evidence=[dict(id='s')])])
        for encoder,index,stage in ((FailedEncoder(),Index(),'encoding'),(InvalidEncoder(),Index(),'encoding'),(Encoder(),FailedIndex(),'retrieval'),(Encoder(),DuplicateIndex(),'retrieval')):
            with self.subTest(stage=stage,encoder=type(encoder).__name__,index=type(index).__name__):
                assessor=Assessor();events=GroundedPipeline(encoder,index,Catalog(),Planner(),assessor).run('Find a method.',k=1,candidates=2)
                self.assertEqual(events[-1]['stage'],stage);self.assertEqual(events[-1]['status'],'unresolved');self.assertEqual(assessor.calls,[])

    def test_missing_evidence_and_candidate_shortfall_are_not_complete(self):
        class EmptyEvidence:
            def search_vector(self,q,k):return dict(hits=[dict(id='a',evidence=[])])
        assessor=Assessor();events=GroundedPipeline(Encoder(),EmptyEvidence(),Catalog(),Planner(),assessor).run('Find a method.',k=1,candidates=2)
        self.assertEqual(events[-1]['status'],'partial');self.assertEqual(events[-1]['unresolved_experts'],['a']);self.assertEqual(assessor.calls,[])
        class ShortIndex:
            def search_vector(self,q,k):return dict(hits=[dict(id='a',evidence=[dict(id='s')])])
        events=GroundedPipeline(Encoder(),ShortIndex(),Catalog(),Planner(),Assessor()).run('Find a method.',k=1,candidates=2)
        self.assertEqual(events[-1]['status'],'partial');self.assertEqual(events[-1]['candidate_count'],1)

    def test_invalid_plan_and_boolean_budgets_are_rejected(self):
        class Malformed:
            def plan(self,demand):return {}
        pipeline=GroundedPipeline(Encoder(),Index(),Catalog(),Malformed(),Assessor())
        self.assertEqual(pipeline.run('Find a method.')[0]['status'],'unresolved')
        with self.assertRaises(ValueError):pipeline.run('Find a method.',k=True,candidates=2)

    def test_planning_failure_never_silently_becomes_empty_success(self):
        class Broken:
            def plan(self,demand):raise TimeoutError()
        events=GroundedPipeline(Encoder(),Index(),Catalog(),Broken(),Assessor()).run('Find a method.')
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['status'],'unresolved')
