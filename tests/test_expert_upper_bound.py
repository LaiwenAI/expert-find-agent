import unittest
import numpy as np
from expert_search.expert_upper_bound import ExpertTermUpperBounds


class Resident:
    ids=('s1','s2','s3')
    indices=np.array([0,1,2,0,1,2],dtype=np.int32)
    values=np.array([5.,0.,4.,0.,5.,4.])
    offsets={'alpha':(0,3),'beta':(3,6)}


class ExpertUpperBoundTest(unittest.TestCase):
    def test_maximum_across_distinct_sources_is_safe(self):
        bounds=ExpertTermUpperBounds(Resident(),{'a':['s1','s2'],'b':['s3']})
        values=bounds.upper('alpha beta',['a','b'])
        self.assertGreaterEqual(values['a'],10.)
        self.assertGreaterEqual(values['b'],8.)
        self.assertGreater(values['a'],5.)  # Each source of a scores only five.

    def test_bad_ownership_or_query(self):
        with self.assertRaises(ValueError):ExpertTermUpperBounds(Resident(),{'a':['s1'],'b':['s1','s3']})
        bounds=ExpertTermUpperBounds(Resident(),{'a':['s1','s2'],'b':['s3']})
        with self.assertRaises(ValueError):bounds.upper('alpha',['unknown'])
        with self.assertRaises(ValueError):bounds.upper('alpha',['a','a'])


if __name__=='__main__':unittest.main()

class PipelineBoundTest(unittest.TestCase):
    def test_pruning_and_cross_source_bound(self):
        from expert_search.expert_upper_bound import certified_candidate_max
        class FakeResident(Resident):
            def selected_scores(self,query,ids):
                terms=set(query.split());scores={}
                for source in ids:
                    index=self.ids.index(source);scores[source]=sum(self.values[lo+index] for term,(lo,_) in self.offsets.items() if term in terms)
                return scores
        class Ranker:
            def search(self,query,k):
                return {'results':[{'expert':{'id':e},'score':0} for e in ('a','b')], 'source_snapshot_sha256':'fixed'}
        class Evidence:
            def evidence_from_scores(self,ranking,owners,scores,k):
                return {e:[{'id':s,'score':scores[s]} for s in owners[e]][:k] for e in ranking}
        class Pipeline:
            def __init__(self):
                self.resident=FakeResident();self.ranker=Ranker();self.evidence=Evidence();self.evidence_by_expert={'a':['s1','s2'],'b':['s3']}
            def _check_resident_snapshot(self):pass
        pipeline=Pipeline();bounds=ExpertTermUpperBounds(pipeline.resident,pipeline.evidence_by_expert)
        result=certified_candidate_max(pipeline,bounds,'alpha',k=1,pool=2,seed=1)
        self.assertEqual(result['results'][0]['expert']['id'],'a');self.assertEqual(result['pruned_candidates'],1)
        result=certified_candidate_max(pipeline,bounds,'alpha beta',k=1,pool=2,seed=1)
        self.assertEqual(result['results'][0]['expert']['id'],'b');self.assertEqual(result['pruned_candidates'],0)
        pipeline.evidence_by_expert['a']=['s2','s1']
        with self.assertRaises(ValueError):certified_candidate_max(pipeline,bounds,'alpha',k=1,pool=2,seed=1)
