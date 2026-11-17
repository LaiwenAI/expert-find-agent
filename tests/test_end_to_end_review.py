import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.retrieval.check_end_to_end_review import check


class EndToEndReviewTest(unittest.TestCase):
    def test_export_hides_model_output_and_preserves_visible_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();out=root/'review'
            (run/'protocol.json').write_text(json.dumps({'query_ids':['q'],'arms':['single_pass','tree_agent']}))
            result={'query':'need','results':[{'expert':{'id':'expert'},'reason':'model explanation',
                    'evidence':[{'id':'source','expert_id':'expert','text':'a'*7000}]}]}
            (run/'q-single_pass.json').write_text(json.dumps({'query_id':'q','arm':'single_pass','status':'ok','result':result}))
            (run/'q-tree_agent.json').write_text(json.dumps({'query_id':'q','arm':'tree_agent','status':'error'}))
            subprocess.run([sys.executable,'-m','experiments.retrieval.export_end_to_end_review',str(run),str(out)],check=True,capture_output=True)
            form=json.loads((out/'blind-review.jsonl').read_text());self.assertEqual(len(form['evidence'][0]['text']),6000)
            self.assertNotIn('arm',form);self.assertNotIn('rank',form);self.assertNotIn('expert',form)
            self.assertNotIn('model explanation',json.dumps(form));self.assertEqual(check(out,out/'blind-review.jsonl')['evidence_fit'],{'missing':1})

    def test_rejects_changed_evidence_and_labels_without_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            row={'case_id':'case-001','original_need':'need','evidence':[{'id':'source-1','text':'evidence'}],
                 'evidence_fit':None,'core_capability_support':None,'rationale':None,'reviewer_id':None}
            payload=(json.dumps(row)+'\n').encode();(root/'blind-review.jsonl').write_bytes(payload)
            (root/'manifest.json').write_text(json.dumps({'cases':1,'review_sha256':hashlib.sha256(payload).hexdigest()}))
            review=root/'review.jsonl';review.write_bytes(payload)
            self.assertEqual(check(root,review)['evidence_fit'],{'missing':1})
            row['evidence_fit']='supported';review.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):check(root,review)
            row.update(reviewer_id='reviewer',rationale='Matches the need')
            review.write_text(json.dumps(row)+'\n')
            self.assertEqual(check(root,review)['evidence_fit'],{'supported':1})
            row['evidence'][0]['text']='changed';review.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):check(root,review)


if __name__=='__main__':unittest.main()
