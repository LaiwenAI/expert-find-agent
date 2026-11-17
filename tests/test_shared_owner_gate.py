import unittest
import numpy as np
from expert_search.shared_owner_gate import SharedOwnerGate


class Resident:
    ids=('shared','a','b')
    indices=np.array([0,1,2,0,1,2],dtype=np.int32)
    values=np.array([1.,1.,2.,1.,2.,1.])
    offsets={'alpha':(0,3),'beta':(3,6)}
    def selected_scores(self,query,ids):
        terms=set(query.split());return {sid:sum(self.values[lo+self.ids.index(sid)] for term,(lo,_) in self.offsets.items() if term in terms) for sid in ids}


class SharedOwnerGateTest(unittest.TestCase):
    def test_shared_sources_ties_and_full_ranking(self):
        index=Resident();owners={'a':['shared','a'],'b':['shared','b'],'c':['shared']}
        gate=SharedOwnerGate(index,owners)
        for query in ('alpha','beta','alpha beta','missing'):
            scores=index.selected_scores(query,index.ids)
            ranking=sorted(owners,key=lambda e:(-max(scores[s] for s in owners[e]),e))
            for k in (1,2,3):
                for e in owners:
                    expected=e in ranking[:k] and max(scores[s] for s in owners[e])>0
                    self.assertEqual(gate.contains(query,e,k)['accepted'],expected,(query,k,e))

    def test_random_shared_ownership_matches_exhaustive(self):
        import random
        rng=random.Random(91)
        for _ in range(50):
            class Sample:
                ids=('s0','s1','s2','s3')
                indices=np.array([0,1,2,3,0,1,2,3],dtype=np.int32)
                values=np.array([rng.choice((0.,.1,1.,1.0000000000000002)) for _ in range(8)])
                offsets={'alpha':(0,4),'beta':(4,8)}
                def selected_scores(self,query,ids):
                    return {sid:sum(self.values[lo+self.ids.index(sid)] for term,(lo,_) in self.offsets.items() if term in query.split()) for sid in ids}
            owner={'a':['s0','s1'],'b':['s1','s2'],'c':['s2','s3'],'d':['s0','s3']}
            index=Sample();gate=SharedOwnerGate(index,owner)
            for query in ('alpha','beta','alpha beta'):
                scores=index.selected_scores(query,index.ids)
                order=sorted(owner,key=lambda e:(-max(scores[s] for s in owner[e]),e))
                for k in (1,2,4):
                    batched={r['candidate']:r for r in gate.contains_many(query,list(owner),k)}
                    for e in owner:
                        expected=e in order[:k] and max(scores[s] for s in owner[e])>0
                        self.assertEqual(gate.contains(query,e,k)['accepted'],expected)
                        self.assertEqual(batched[e]['accepted'],expected)

    def test_batched_matches_individual_with_shared_owners(self):
        gate=SharedOwnerGate(Resident(),{'a':['shared','a'],'b':['shared','b'],'c':['shared']})
        for query in ('alpha','beta','alpha beta','missing'):
            for k in (1,2,3):
                batch=gate.contains_many(query,['c','a','b'],k)
                for row in batch:
                    single=gate.contains(query,row['candidate'],k)
                    self.assertEqual(row['accepted'],single['accepted'])
                    self.assertEqual(row['candidate_score'],single['candidate_score'])
        with self.assertRaises(ValueError):gate.contains_many('alpha',['a','a'],1)

    def test_invalid_and_negative_postings_rejected(self):
        index=Resident();index.values=index.values.copy();index.values[0]=-1
        with self.assertRaises(ValueError):SharedOwnerGate(index,{'a':['shared','a','b']})
        with self.assertRaises(ValueError):SharedOwnerGate(Resident(),{'a':['shared','a']})
        gate=SharedOwnerGate(Resident(),{'a':['shared','a'],'b':['b']})
        with self.assertRaises(ValueError):gate.contains('alpha','missing',1)

if __name__=='__main__':unittest.main()
