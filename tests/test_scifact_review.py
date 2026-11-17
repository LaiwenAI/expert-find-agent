import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from experiments.scifact.check_unannotated_review import check
from experiments.scifact.decisions_oracle_pilot import sha


class SciFactReviewTest(unittest.TestCase):
    def test_missing_exact_quote_and_changed_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);view={'case_id':'case-001','claim':'A claim','title':'A title','abstract':['A source sentence.']}
            digest=hashlib.sha256(json.dumps(view,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
            row=dict(**view,view_sha256=digest,verdict=None,evidence_quote=None,reviewer_id=None,notes=None)
            (root/'private-mapping.json').write_text(json.dumps([{'case_id':'case-001','view_sha256':digest}]))
            (root/'manifest.json').write_text(json.dumps({'cases':1,'mapping_sha256':sha(root/'private-mapping.json')}))
            review=root/'review.json';review.write_text(json.dumps([row]))
            self.assertEqual(check(root,review)['counts'],{'missing':1})
            row.update(verdict='support',evidence_quote='source sentence',reviewer_id='reviewer')
            review.write_text(json.dumps([row]));self.assertEqual(check(root,review)['counts'],{'support':1})
            row['evidence_quote']='fabricated';review.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):check(root,review)
            row['evidence_quote']='source sentence';row['claim']='Changed';review.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):check(root,review)


if __name__=='__main__':unittest.main()
