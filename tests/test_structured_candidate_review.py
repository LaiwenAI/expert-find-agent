import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.retrieval.check_structured_candidate_review import check


class StructuredReviewTest(unittest.TestCase):
    def test_missing_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'blind-review.jsonl';mapping=root/'private-mapping.json';review=root/'review.jsonl'
            row={'case_id':'x','source_views':[{'text':'method'}],'historical_support':None,'target_alignment':None,'bridge_validity':None,'personal_contribution':None,'rationale':None,'reviewer_id':None}
            base.write_text(json.dumps(row)+'\n');mapping.write_text('[]');digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'cases':1,'review_sha256':digest(base),'mapping_sha256':digest(mapping)}))
            review.write_text(json.dumps(row)+'\n');self.assertEqual(check(root,review)['counts']['bridge_validity']['missing'],1)
            row['bridge_validity']='supported';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Reviewer'):check(root,review)
            row['reviewer_id']='R';row['rationale']='reason';row['source_views'][0]['text']='modified';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Changed source'):check(root,review)


if __name__=='__main__':unittest.main()
