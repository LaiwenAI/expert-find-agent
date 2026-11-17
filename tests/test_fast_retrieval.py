import json
import tempfile
import unittest
from pathlib import Path
from expert_search.retrieval import Search,index
from expert_search.fast_retrieval import FastSearch,compile_index

class FastTest(unittest.TestCase):
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
            self.assertTrue(all(e['expert_id']=='b' for e in selected['b']))
            self.assertAlmostEqual(fast.document_scores('biology')['7'],fast.search('biology',1)[0]['evidence'][0]['score'])
            # With no lexical overlap, fallback ordering is deterministic and
            # never borrows another expert's evidence.
            fallback=fast.evidence_for('unmatched',['a'],2)
            self.assertEqual([e['id'] for e in fallback['a']],['0','1'])
            with self.assertRaises(ValueError):fast.evidence_for('biology',['missing'])
