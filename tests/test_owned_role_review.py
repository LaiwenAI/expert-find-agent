import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.retrieval.check_owned_role_review import check


class OwnedRoleReviewTest(unittest.TestCase):
    def test_missing_mixed_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row={'case_id':'case-001','source_text':'abc','windows':[{'id':0,'start':0,'end':3,'text':'abc','label':None,'reason':None}],'reviewer_id':None}
            payload=(json.dumps(row)+'\n').encode();(root/'blind-review.jsonl').write_bytes(payload)
            (root/'manifest.json').write_text(json.dumps({'review_sha256':hashlib.sha256(payload).hexdigest(),'cases':1,'windows':1}))
            review=root/'review.jsonl';review.write_bytes(payload);self.assertEqual(check(root,review),{'missing':1})
            row['reviewer_id']='reviewer';row['windows'][0].update(label='mixed',reason='Multiple roles')
            review.write_text(json.dumps(row)+'\n');self.assertEqual(check(root,review),{'mixed':1})
            row['windows'][0]['text']='changed';review.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):check(root,review)


if __name__=='__main__':unittest.main()
