import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from experiments.retrieval.check_joint_review import check


def write(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class JointReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.row={'case_id':'case-1','need':'need','sources':[{'id':'source-1','kind':'paper','text':'evidence'}],
                  'method_fit':None,'constraint_support':None,'personal_contribution':None,'rationale':None,'reviewer_id':None}
        mapping={'case_id':'case-1','query_id':'q','expert_id':'e','sources':[{'alias':'source-1','id':'s','start':0,'end':8,'sha256':hashlib.sha256(b'evidence').hexdigest()}]}
        write(self.root/'blind-review.jsonl',[self.row]);write(self.root/'private-mapping.jsonl',[mapping])
        manifest={'candidate_views':1,'outputs_sha256':{n:sha(self.root/n) for n in ('blind-review.jsonl','private-mapping.jsonl')}}
        (self.root/'manifest.json').write_text(json.dumps(manifest));self.review=self.root/'review.jsonl'

    def tearDown(self):self.temp.cleanup()

    def test_missing_and_valid_label_are_distinct(self):
        write(self.review,[self.row]);self.assertEqual(check(self.root,self.review)['counts']['method_fit'],{'missing':1})
        row={**self.row,'method_fit':'not_supported_by_view','rationale':'No method shown','reviewer_id':'r1'}
        write(self.review,[row]);self.assertEqual(check(self.root,self.review)['counts']['method_fit'],{'not_supported_by_view':1})

    def test_changed_evidence_and_unattributed_labels_rejected(self):
        row={**self.row,'need':'changed'};write(self.review,[row])
        with self.assertRaisesRegex(ValueError,'Changed need'):check(self.root,self.review)
        row={**self.row,'method_fit':'supported'};write(self.review,[row])
        with self.assertRaisesRegex(ValueError,'reviewer'):check(self.root,self.review)

if __name__=='__main__':unittest.main()
