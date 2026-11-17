import json
import tempfile
import unittest
from pathlib import Path

from expert_search.fast_retrieval import FastSearch, compile_index
from expert_search.resident_candidate_bm25 import ResidentCandidateBM25
from expert_search.retrieval import index


class ResidentCandidateBM25Test(unittest.TestCase):
    def test_exact_scores_for_subsets_and_empty_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experts.jsonl').write_text(json.dumps({'id': 'a'}) + '\n')
            evidence = [{'id': str(i), 'expert_id': 'a', 'text': text} for i, text in enumerate(['alpha beta', 'alpha alpha', '\u673a\u5668\u4eba beta', 'other'])]
            (root / 'evidence.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in evidence))
            index(root, root / 'source.sqlite'); compile_index(root / 'source.sqlite', root / 'fast.sqlite')
            sqlite = FastSearch(root, root / 'fast.sqlite'); resident = ResidentCandidateBM25(root / 'fast.sqlite'); adaptive = ResidentCandidateBM25(root / 'fast.sqlite', adaptive_intersection=True)
            for query in ('alpha beta', 'unmatched', '\u673a\u5668\u4eba alpha', ''):
                for ids in ([], ['3','1','0'], ['2','2','0']):
                    self.assertEqual(resident.selected_scores(query, ids), sqlite.filtered_global_document_scores(query, ids))
                    self.assertEqual(adaptive.selected_scores(query, ids), resident.selected_scores(query, ids))
                    self.assertEqual(adaptive.selected_scores_batched(query, ids, batch_limit=2), resident.selected_scores(query, ids))
            with self.assertRaises(ValueError): resident.selected_scores('alpha', ['missing'])
            with self.assertRaises(ValueError): adaptive.selected_scores('alpha', ['missing'])
            sqlite.db.close()


    def test_randomized_intersection_directions_match_bitwise(self):
        import numpy as np
        rng=np.random.default_rng(20261122);resident=ResidentCandidateBM25.__new__(ResidentCandidateBM25)
        resident.ids=tuple(f's{i}' for i in range(80));resident.evidence_positions={e:i for i,e in enumerate(resident.ids)}
        indices=[];values=[];offsets={}
        for term,count in [('alpha',2),('beta',17),('gamma',79),('delta',41)]:
            selected=np.sort(rng.choice(80,count,replace=False));lo=len(indices);indices.extend(selected);values.extend(rng.random(count));offsets[term]=(lo,len(indices))
        resident.indices=np.array(indices,dtype=np.int32);resident.values=np.array(values,dtype=np.float64);resident.offsets=offsets
        for count in (0,1,4,20,79,80):
            for _ in range(10):
                ids=[resident.ids[i] for i in rng.choice(80,count,replace=False)]
                resident.adaptive_intersection=False;expected=resident.selected_scores('alpha beta gamma delta',ids)
                resident.adaptive_intersection=True;actual=resident.selected_scores('alpha beta gamma delta',ids)
                self.assertEqual(actual,expected)
                self.assertEqual(resident.selected_scores_batched('alpha beta gamma delta',ids,batch_limit=10),expected)


if __name__ == '__main__':
    unittest.main()
