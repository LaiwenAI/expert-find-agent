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
            sqlite = FastSearch(root, root / 'fast.sqlite'); resident = ResidentCandidateBM25(root / 'fast.sqlite')
            for query in ('alpha beta', 'unmatched', '\u673a\u5668\u4eba alpha', ''):
                for ids in ([], ['3','1','0'], ['2','2','0']):
                    self.assertEqual(resident.selected_scores(query, ids), sqlite.filtered_global_document_scores(query, ids))
            with self.assertRaises(ValueError): resident.selected_scores('alpha', ['missing'])
            sqlite.db.close()


if __name__ == '__main__':
    unittest.main()
