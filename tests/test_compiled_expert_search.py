import json
import tempfile
import unittest
from pathlib import Path
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.compiled_expert_search import CompiledExpertSearch,corpus_snapshot


class CompiledExpertSearchTest(unittest.TestCase):
    def test_persistent_query_matches_reference_and_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = {'a':['d1'],'b':['d2']}
            texts = {'d1':'alpha alpha beta','d2':'gamma beta'}
            (root/'experts.jsonl').write_text(''.join(json.dumps({'id':e})+'\n' for e in profiles))
            (root/'evidence.jsonl').write_text(''.join(json.dumps({'id':p,'text':t})+'\n' for p,t in texts.items()))
            reference = ExpertLanguageModels(texts,profiles)
            CompiledCandidateLM(reference).save(root/'index',corpus_snapshot(root))
            engine = CompiledExpertSearch(root,root/'index')
            result = engine.search('alpha',2)
            expected = reference.score('alpha')
            self.assertEqual([r['expert']['id'] for r in result['results']],sorted(expected,key=lambda e:(-expected[e],e)))
            self.assertEqual([r['score'] for r in result['results']],[expected[r['expert']['id']] for r in result['results']])
            with self.assertRaises(ValueError):
                engine.search('alpha',True)
            (root/'evidence.jsonl').write_text('changed corpus')
            with self.assertRaises(ValueError):
                CompiledExpertSearch(root,root/'index')
