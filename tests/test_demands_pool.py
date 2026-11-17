import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from expert_search.demands import pool


class FakeSearch:
    def __init__(self,*args):pass
    def search(self,query,k,method):
        sources={'bm25-max':['a','shared'],'bm25-vote':['shared'],'tfidf-max':['b','shared']}[method]
        return [{'id':'person','name':'Name','evidence':[{'id':sid,'expert_id':'person','text':sid,'score':9.0} for sid in sources]}]


class DemandsPoolTest(unittest.TestCase):
    def test_union_deduplicates_and_hides_scores(self):
        with tempfile.TemporaryDirectory() as temp, patch('expert_search.demands.Search',FakeSearch):
            report=pool([{'id':'q','query':'test'}],output=temp)
            row=json.loads((Path(temp)/'annotation-pool.jsonl').read_text())
            self.assertEqual(report['annotation_pairs'],1)
            self.assertEqual([x['id'] for x in row['evidence']],['a','b','shared'])
            self.assertTrue(all('score' not in x for x in row['evidence']))
            self.assertEqual(len((Path(temp)/'baseline-runs.jsonl').read_text().splitlines()),3)

    def test_owner_mismatch_is_rejected(self):
        class Bad(FakeSearch):
            def search(self,*args):
                rows=super().search(*args);rows[0]['evidence'][0]['expert_id']='other';return rows
        with tempfile.TemporaryDirectory() as temp, patch('expert_search.demands.Search',Bad):
            with self.assertRaisesRegex(ValueError,'owner'):pool([{'id':'q','query':'test'}],output=temp)

if __name__=='__main__':unittest.main()
