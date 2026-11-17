import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.check_prioritized_fresh_review import check


class PrioritizedFreshReviewTest(unittest.TestCase):
    def test_content_and_positive_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'snapshot_id': 'case-001', 'query': 'need', 'source_view': [{'id': 's1', 'text': 'evidence'}],
                   'annotation': {'grade': None, 'covered_operation': None, 'source_citations': None, 'mechanism': None,
                                  'open_assumptions': None, 'remaining_scope': None, 'source_owner_verified': None, 'annotator': None}}
            baseline = root / 'blind-review.jsonl'; baseline.write_text(json.dumps(row) + '\n')
            mapping = root / 'private-mapping.json'; mapping.write_text('[]')
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps({'cases': 1, 'blind_sha256': sha(baseline), 'mapping_sha256': sha(mapping)}))
            review = root / 'review.jsonl'; review.write_text(json.dumps(row) + '\n')
            self.assertEqual(check(root, review)['grades'], {'missing': 1})
            row['annotation'].update(grade=2, annotator='r', source_citations=['s1'], covered_operation='operation', mechanism='method', open_assumptions='validation')
            review.write_text(json.dumps(row) + '\n'); self.assertEqual(check(root, review)['grades'], {'2': 1})
            row['annotation']['source_citations'] = ['foreign']; review.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError): check(root, review)
            row['query'] = 'changed'; review.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError): check(root, review)


if __name__ == '__main__':
    unittest.main()
