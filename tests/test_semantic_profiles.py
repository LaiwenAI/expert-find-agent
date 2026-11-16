import unittest
import numpy as np
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.semantic_profiles import TransportedProfiles,mutual_kernel


class SemanticProfilesTest(unittest.TestCase):
    def test_identity_kernel_and_compilation_preserve_scores(self):
        original=ExpertLanguageModels({'d1':'alpha alpha beta','d2':'beta gamma'}, {'a':['d1'],'b':['d2']})
        identity=TransportedProfiles(original,{})
        self.assertEqual(original.score('alpha gamma'),identity.score('alpha gamma'))
        semantic=TransportedProfiles(original,{'alpha':{'alpha':.5,'gamma':.5},'gamma':{'gamma':.5,'alpha':.5}})
        self.assertAlmostEqual(sum(semantic.background.values()),1)
        for distribution in semantic.profile_terms.values():self.assertAlmostEqual(sum(distribution.values()),1)
        self.assertEqual(semantic.score('alpha beta'),CompiledCandidateLM(semantic).score('alpha beta'))

    def test_kernel_validation_and_mutual_graph(self):
        vectors=np.array([[1.,0.],[.99,.1],[0.,1.]])
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        kernel=mutual_kernel(['alpha','beta','gamma'],vectors,k=1)
        self.assertEqual(kernel['gamma'],{'gamma':1.})
        self.assertAlmostEqual(sum(kernel['alpha'].values()),1)
        original=ExpertLanguageModels({'d':'alpha beta'},{'e':['d']})
        with self.assertRaises(ValueError):TransportedProfiles(original,{'alpha':{'alpha':.8}})
