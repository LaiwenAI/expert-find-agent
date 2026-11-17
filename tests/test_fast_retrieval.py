import json
import tempfile
import unittest
from pathlib import Path
from expert_search.retrieval import Search,index
from expert_search.fast_retrieval import FastSearch,compile_index,add_selective_indexes
from expert_search.agent import verify_compiled_documents
from experiments.retrieval.audit_compiled_index import audit

class FastTest(unittest.TestCase):
    def test_batched_lookup_crosses_sql_parameter_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'experts.jsonl').write_text('\n'.join(json.dumps({'id':x}) for x in ['a','b']))
            evidence=[{'id':str(i),'expert_id':'a' if i%2 else 'b','text':'robot control'} for i in range(1001)]
            (p/'evidence.jsonl').write_text('\n'.join(json.dumps(x) for x in evidence))
            index(p,p/'old');compile_index(p/'old',p/'fast')
            verify_compiled_documents(p/'old',p/'fast')
            old=Search(p,p/'old');fast=FastSearch(p,p/'fast')
            self.assertGreater(audit(p/'old',p/'fast')['postings_verified'],0)
            add_selective_indexes(p/'fast')
            ids=[str(i) for i in range(1001)]
            query=' '.join('term'+str(i) for i in range(450))+' robot control'
            full=fast.document_scores(query)
            self.assertEqual(fast.selective_document_scores(query,ids),{eid:full.get(eid,0.) for eid in ids})
            for method in ['bm25-max','bm25-vote','tfidf-max']:
                a=old.search('robot control',2,method,['0','1']);b=fast.search('robot control',2,method,['0','1'])
                self.assertEqual([r['id'] for r in a],[r['id'] for r in b])
                for x,y in zip(a,b):
                    self.assertAlmostEqual(x['score'],y['score'])
                    self.assertEqual([e['id'] for e in x['evidence']],[e['id'] for e in y['evidence']])
            import sqlite3
            with sqlite3.connect(p/'fast') as db:
                db.execute("UPDATE docs SET body='{}' WHERE id=0")
            with self.assertRaises(ValueError):
                verify_compiled_documents(p/'old',p/'fast')

    def test_exact_rank_and_scores(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'experts.jsonl').write_text('\n'.join(json.dumps({'id':x}) for x in ['a','b']))
            evidence=[{'id':str(i),'expert_id':'a' if i<6 else 'b','text':t} for i,t in enumerate(['robot control']*6+['\u673a\u5668\u4eba control','biology'])]
            (p/'evidence.jsonl').write_text('\n'.join(json.dumps(x) for x in evidence))
            index(p,p/'old');compile_index(p/'old',p/'fast')
            old=Search(p,p/'old');fast=FastSearch(p,p/'fast')
            for method in ['bm25-max','bm25-vote','tfidf-max']:
                for excluded in [[],['0','1'],[str(i) for i in range(8)]]:
                    a=old.search('robot \u673a\u5668\u4eba control',2,method,excluded);b=fast.search('robot \u673a\u5668\u4eba control',2,method,excluded)
                    self.assertEqual([r['id'] for r in a],[r['id'] for r in b])
                    for x,y in zip(a,b):
                        self.assertAlmostEqual(x['score'],y['score'])
                        self.assertEqual([e['id'] for e in x['evidence']],[e['id'] for e in y['evidence']])
            selected=fast.evidence_for('biology',['b'],1)
            self.assertEqual([e['id'] for e in selected['b']],['7'])
            add_selective_indexes(p/'fast')
            long_query=' '.join('term'+str(i) for i in range(450))+' robot control'
            for query in ['robot \u673a\u5668\u4eba control','unmatched','biology',long_query]:
                self.assertEqual(fast.evidence_for(query,['a','b'],3),fast.evidence_for_selective(query,['a','b'],3))
            for query in ['robot \u673a\u5668\u4eba control','unmatched',long_query]:
                full=fast.document_scores(query)
                chosen=[str(i) for i in range(8)]
                self.assertEqual(fast.selective_document_scores(query,chosen),{eid:full.get(eid,0.) for eid in chosen})
            self.assertEqual(fast.selective_document_scores('biology',[]),{})
            with self.assertRaises(ValueError):fast.selective_document_scores('biology',['missing'])
            self.assertEqual(fast.evidence_for_selective('biology',[],3),{})
            with self.assertRaises(ValueError):fast.evidence_for_selective('biology',['missing'])
            self.assertTrue(all(e['expert_id']=='b' for e in selected['b']))
            self.assertAlmostEqual(fast.document_scores('biology')['7'],fast.search('biology',1)[0]['evidence'][0]['score'])
            # With no lexical overlap, fallback ordering is deterministic and
            # never borrows another expert's evidence.
            fallback=fast.evidence_for('unmatched',['a'],2)
            self.assertEqual([e['id'] for e in fallback['a']],['0','1'])
            with self.assertRaises(ValueError):fast.evidence_for('biology',['missing'])
            import sqlite3
            with sqlite3.connect(p/'fast') as db:
                db.execute('UPDATE scores SET bm25=bm25+1 WHERE rowid=(SELECT min(rowid) FROM scores)')
            with self.assertRaises(ValueError):audit(p/'old',p/'fast')
