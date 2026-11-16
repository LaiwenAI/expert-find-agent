import random
import unittest
import numpy as np
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.certified_topk import certified_topk


class CertifiedTopKTest(unittest.TestCase):
    def test_randomized_reference_order_scores_and_eligible(self):
        rng=random.Random(31)
        words=['alpha','beta','gamma','delta','epsilon']
        docs={str(i):' '.join(rng.choices(words,k=rng.randint(1,70))) for i in range(40)}
        profiles={f'e{i:02}':rng.sample(list(docs),rng.randint(1,5)) for i in range(30)}
        profiles['duplicate']=profiles['e00']
        for smoothing in (.01,.5,1.):
            reference=ExpertLanguageModels(docs,profiles,smoothing)
            model=CompiledCandidateLM(reference)
            for _ in range(30):
                query=' '.join(rng.choices(words+['oov'],k=rng.randint(1,100)))
                eligible=rng.sample(sorted(model.available),rng.randint(1,len(model.available)))
                scores=reference.score(query,eligible=eligible)
                for k in (1,5,50):
                    expected=sorted(scores,key=lambda e:(-scores[e],e))[:k]
                    actual=certified_topk(model,query,k,eligible)
                    self.assertEqual(actual['results'],[dict(id=e,score=scores[e]) for e in expected])

    def test_zero_ties_empty_and_input_rejection(self):
        model=CompiledCandidateLM(ExpertLanguageModels({'d':'alpha beta'},{'z':['d'],'a':['d']}))
        self.assertEqual(certified_topk(model,'oov',1)['results'],[dict(id='a',score=0.)])
        self.assertEqual(certified_topk(model,'oov',1)['exact_candidates'],1)
        self.assertEqual(certified_topk(model,'alpha',3,[])['results'],[])
        for k in (0,True,1.5):
            with self.assertRaises(ValueError):certified_topk(model,'alpha',k)
        for eligible in (['a','a'],['missing']):
            with self.assertRaises(ValueError):certified_topk(model,'alpha',1,eligible)

    def test_rounded_sums_across_extreme_magnitudes_and_near_ties(self):
        model=CompiledCandidateLM(ExpertLanguageModels({'d':'alpha beta gamma'},
            {f'e{i:02}':['d'] for i in range(20)}))
        rng=random.Random(92)
        for _ in range(50):
            for term in model.postings:
                indices,values=model.postings[term]
                model.postings[term]=(indices,np.asarray([10.**rng.randint(-200,200) for _ in values]))
            query='alpha '*50+'beta '*23+'gamma '
            scores=model.score(query)
            for k in (1,5,20):
                expected=sorted(scores,key=lambda e:(-scores[e],e))[:k]
                self.assertEqual(certified_topk(model,query,k)['results'],
                    [dict(id=e,score=scores[e]) for e in expected])
