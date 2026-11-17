import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.retrieval.check_enterprise_adaptive_review import check


class EnterpriseAdaptiveReviewTest(unittest.TestCase):
    def test_partial_review_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(case_id=str(i), query='query', reason='reason', missing=[], sources=[],
                         factual_support=None, query_relevance=None, personal_role_support=None,
                         review_notes=None, reviewer_id=None) for i in range(3)]
            mapping = [dict(case_id=str(i), arm='lm' if i < 2 else 'adaptive_bm25') for i in range(3)]
            baseline = root / 'blind-review.jsonl'
            baseline.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            mapping_path = root / 'private-mapping.json'; mapping_path.write_text(json.dumps(mapping))
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(dict(cases=3, blind_review_sha256=sha(baseline), private_mapping_sha256=sha(mapping_path))))
            rows[0].update(factual_support='supported', query_relevance='unsupported',
                           personal_role_support='unknown', review_notes='reviewed', reviewer_id='r1')
            review = root / 'review.jsonl'; review.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = check(root, review)
            self.assertEqual(result['arms']['lm']['factual_support']['supported_fraction_bounds'], [.5, 1.])
            self.assertEqual(result['arms']['lm']['query_relevance']['supported_fraction_bounds'], [0., .5])
            self.assertEqual(result['arms']['lm']['personal_role_support']['supported_fraction_bounds'], [0., 1.])
            rows[1]['query'] = 'changed'
            review.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(ValueError):
                check(root, review)
            mapping_path.write_text('[]')
            with self.assertRaises(ValueError):
                check(root, baseline)


if __name__ == '__main__':
    unittest.main()
