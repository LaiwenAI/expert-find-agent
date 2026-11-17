import json
import tempfile
import unittest
from pathlib import Path
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.compiled_expert_search import CompiledExpertSearch,corpus_snapshot
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.fast_retrieval import add_selective_indexes,compile_index
from expert_search.retrieval import index
from experiments.retrieval.build_candidate_pipeline import build


class CompiledExpertSearchTest(unittest.TestCase):
    def test_builder_creates_a_working_pipeline_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'experts.jsonl').write_text(json.dumps({'id':'a'})+'\n')
            (root/'evidence.jsonl').write_text(json.dumps({'id':'d','expert_id':'a','text':'robot control'})+'\n')
            index(root,root/'source.sqlite')
            result=build(root,root/'source.sqlite',root/'lm',root/'fast.sqlite')
            self.assertEqual(result['documents'],1)
            self.assertEqual(CandidateEvidencePipeline(root,root/'lm',root/'fast.sqlite').search('robot',1)['results'][0]['expert']['id'],'a')
            with self.assertRaises(FileExistsError):build(root,root/'source.sqlite',root/'lm',root/'fast.sqlite')

    def test_pipeline_binds_candidate_evidence_to_the_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);profiles={'a':['d1'],'b':['d2']};texts={'d1':'alpha alpha','d2':'beta'}
            (root/'experts.jsonl').write_text(''.join(json.dumps({'id':e})+'\n' for e in profiles))
            (root/'evidence.jsonl').write_text(''.join(json.dumps({'id':doc,'expert_id':eid,'text':texts[doc]})+'\n' for eid,docs in profiles.items() for doc in docs))
            CompiledCandidateLM(ExpertLanguageModels(texts,profiles)).save(root/'index',corpus_snapshot(root))
            index(root,root/'source.sqlite');compile_index(root/'source.sqlite',root/'fast.sqlite');add_selective_indexes(root/'fast.sqlite')
            pipeline=CandidateEvidencePipeline(root,root/'index',root/'fast.sqlite')
            result=pipeline.search('alpha',2)
            self.assertEqual([r['expert']['id'] for r in result['results']],['a','b'])
            self.assertTrue(all(e['expert_id']==r['expert']['id'] for r in result['results'] for e in r['evidence']))
            reranked=pipeline.search_reranked('alpha',k=2,candidates=2,evidence_per_expert=1)
            self.assertEqual([r['expert']['id'] for r in reranked['results']],['a','b'])
            self.assertEqual(reranked['results'][1]['score'],0.)
            self.assertTrue(all(e['expert_id']==r['expert']['id'] for r in reranked['results'] for e in r['evidence']))
            with self.assertRaises(ValueError):pipeline.search_reranked('alpha',k=3,candidates=2)
            import sqlite3
            with sqlite3.connect(root/'fast.sqlite') as db:db.execute("UPDATE docs SET expert_id='b' WHERE evidence_id='d1'")
            with self.assertRaises(ValueError):CandidateEvidencePipeline(root,root/'index',root/'fast.sqlite')

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
