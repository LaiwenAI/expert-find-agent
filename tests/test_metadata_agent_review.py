import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.retrieval.check_metadata_agent_review import check


class MetadataAgentReviewTest(unittest.TestCase):
    def test_rejects_altered_sources_and_unattributed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'blind-review.jsonl';mapping=root/'private-mapping.json';review=root/'review.jsonl'
            row={'case_id':'x','original_need':'need','explanation':'claim','cited_sources':[],'explanation_support':None,'individual_contribution':None,'date_representation':None,'rationale':None,'reviewer_id':None}
            base.write_text(json.dumps(row)+'\n');mapping.write_text('[]')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'cases':1,'review_sha256':digest(base),'mapping_sha256':digest(mapping)}))
            review.write_text(json.dumps(row)+'\n');self.assertEqual(check(root,review)['counts']['explanation_support']['missing'],1)
            row['explanation_support']='supported';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Reviewer'):check(root,review)
            row['reviewer_id']='reviewer';row['rationale']='reason';row['explanation']='changed';review.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Changed source'):check(root,review)


if __name__=='__main__':unittest.main()
