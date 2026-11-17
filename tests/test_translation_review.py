import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.bilingual.check_translation_review import check


class TranslationReviewTest(unittest.TestCase):
    def test_reviewer_must_not_change_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row={'case_id':'c','original_need':'need','english_translation':'translated','label':None,'reason':None,'reviewer_id':None}
            payload=(json.dumps(row)+'\n').encode();(root/'blind-review.jsonl').write_bytes(payload)
            (root/'manifest.json').write_text(json.dumps({'cases':1,'review_sha256':hashlib.sha256(payload).hexdigest()}))
            review=root/'review.jsonl';review.write_bytes(payload);self.assertEqual(check(root,review),{'missing':1})
            row.update(label='uncertain',reason='Scope unclear',reviewer_id='r');review.write_text(json.dumps(row)+'\n')
            self.assertEqual(check(root,review),{'uncertain':1})
            row['english_translation']='altered';review.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):check(root,review)


if __name__=='__main__':unittest.main()
