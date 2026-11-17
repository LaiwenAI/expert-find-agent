import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.bilingual.check_grounding_review import check


class GroundingReviewTests(unittest.TestCase):
    def test_missing_labels_and_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, review = root / 'baseline.jsonl', root / 'review.jsonl'
            row = dict(case_id='one', original_need='source', chinese_grounding=None,
                       english_grounding=None, bilingual_equivalence=None,
                       rationale=None, reviewer_id=None)
            payload = (json.dumps(row) + '\n').encode()
            baseline.write_bytes(payload)
            review.write_bytes(payload)
            manifest = dict(output_sha256=hashlib.sha256(payload).hexdigest(), cases=1)
            result = check(baseline, review, manifest)
            self.assertEqual(result['chinese_grounding'], {'missing': 1})
            row['original_need'] = 'changed'
            review.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                check(baseline, review, manifest)

    def test_label_requires_attribution_and_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, review = root / 'baseline.jsonl', root / 'review.jsonl'
            row = dict(case_id='one', chinese_grounding=None, english_grounding=None,
                       bilingual_equivalence=None, rationale=None, reviewer_id=None)
            payload = (json.dumps(row) + '\n').encode()
            baseline.write_bytes(payload)
            row['chinese_grounding'] = 'supported'
            review.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                check(baseline, review, dict(output_sha256=hashlib.sha256(payload).hexdigest(), cases=1))
