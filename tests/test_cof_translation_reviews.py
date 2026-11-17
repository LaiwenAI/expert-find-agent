import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from experiments.bilingual.analyze_cof_translation_reviews import analyze


class TranslationReviewTest(unittest.TestCase):
    def test_missing_uncertainty_and_independent_reviewers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);baseline=root/'blind-review.jsonl';mapping=root/'private-mapping.json';first=root/'first.jsonl';second=root/'second.jsonl'
            row=dict(case_id='c1',english_source='source',chinese_translation='translation',equivalence=None,rationale=None,reviewer_id=None)
            baseline.write_text(json.dumps(row)+'\n');mapping.write_text(json.dumps([dict(case_id='c1')]))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(dict(cases=1,missing_translation_cases=1,review_sha256=sha(baseline),mapping_sha256=sha(mapping))))
            first.write_text(json.dumps(row)+'\n');second.write_text(json.dumps(row)+'\n')
            self.assertIsNone(analyze(root,first,second)['agreement'])
            a=dict(row,equivalence='faithful',rationale='reason',reviewer_id='one');b=dict(row,equivalence='changed',rationale='reason',reviewer_id='two')
            first.write_text(json.dumps(a)+'\n');second.write_text(json.dumps(b)+'\n')
            self.assertEqual(analyze(root,first,second)['counts']['determinate_disagree'],1)
            b['reviewer_id']='one';second.write_text(json.dumps(b)+'\n')
            with self.assertRaises(ValueError):analyze(root,first,second)
            b.update(reviewer_id='two',chinese_translation='altered');second.write_text(json.dumps(b)+'\n')
            with self.assertRaises(ValueError):analyze(root,first,second)


if __name__=='__main__':unittest.main()
