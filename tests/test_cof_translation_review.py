import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.bilingual.check_cof_translation_review import check


class CoFTranslationReviewTest(unittest.TestCase):
    def test_review_requires_immutable_views_and_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);baseline=root/'blind-review.jsonl';mapping=root/'private-mapping.json';review=root/'review.jsonl'
            row={'case_id':'case','english_source':'sample','chinese_translation':'translation','equivalence':None,'rationale':None,'reviewer_id':None}
            baseline.write_text(json.dumps(row)+'\n');mapping.write_text('[]');digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'cases':1,'review_sha256':digest(baseline),'mapping_sha256':digest(mapping)}))
            review.write_text(json.dumps(row)+'\n');self.assertEqual(check(root,review)['counts']['missing'],1)
            row['equivalence']='faithful';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Reviewer'):check(root,review)
            row['reviewer_id']='R';row['rationale']='reason';row['chinese_translation']='changed';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Changed bilingual'):check(root,review)


if __name__=='__main__':unittest.main()
