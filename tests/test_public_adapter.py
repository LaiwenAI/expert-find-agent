import json
import math
import tempfile
import unittest
from pathlib import Path
from expert_search.public_data import adapt
from expert_search.public_benchmark import public_metrics


class AdapterTest(unittest.TestCase):
    def test_graded_gain_matches_trec_linear_convention(self):
        result=public_metrics(['partial','full'],{'full':2,'partial':1})
        expected=(1+2/math.log2(3))/(2+1/math.log2(3))
        self.assertAlmostEqual(result['ndcg@10'],expected)

    def test_logic_type_kept_and_answer_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'qrels').mkdir()
            (root/'corpus.jsonl').write_text(json.dumps({'_id':'d','text':'document'})+'\n')
            (root/'queries.jsonl').write_text(json.dumps({'_id':'q','text':'question','type':'2i','answer':'hidden label'})+'\n')
            (root/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq\td\t2\n')
            adapt(root,root/'adapted')
            row=json.loads((root/'adapted/queries.jsonl').read_text())
            self.assertEqual(row['query_type'],'2i')
            self.assertEqual(row['qrels'],{'d':2})
            self.assertNotIn('answer',row)
