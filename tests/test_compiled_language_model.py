import random
import tempfile
from pathlib import Path
import unittest
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM


class CompiledLanguageTest(unittest.TestCase):
    def test_exact_reference_scores_including_ties_and_oov(self):
        rng = random.Random(12)
        words = ['alpha','beta','gamma','delta','epsilon']
        documents = {str(i):' '.join(rng.choices(words,k=31)) for i in range(20)}
        profiles = {str(i):rng.sample(list(documents),5) for i in range(8)}
        profiles['duplicate'] = profiles['0']
        for smoothing in (.1,.5,1.):
            reference = ExpertLanguageModels(documents,profiles,smoothing)
            compiled = CompiledCandidateLM(reference)
            for _ in range(30):
                query = ' '.join(rng.choices(words+['outsidevocabulary'],k=100))
                self.assertEqual(reference.score(query),compiled.score(query))
                self.assertEqual(reference.score(query,eligible=['0','duplicate']),compiled.score(query,['0','duplicate']))

    def test_persisted_index_scores_and_corruption(self):
        compiled = CompiledCandidateLM(ExpertLanguageModels({'d':'alpha beta'}, {'e':['d']}))
        with tempfile.TemporaryDirectory() as directory:
            compiled.save(directory,'snapshot')
            restored = CompiledCandidateLM.load(directory,'snapshot')
            self.assertEqual(compiled.score('alpha alpha'),restored.score('alpha alpha'))
            with self.assertRaises(ValueError):
                CompiledCandidateLM.load(directory,'different snapshot')
            path = Path(directory)/'arrays.npz'
            path.write_bytes(path.read_bytes()+b'corrupted')
            with self.assertRaises(ValueError):
                CompiledCandidateLM.load(directory,'snapshot')
