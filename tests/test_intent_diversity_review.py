import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from experiments.bilingual.export_intent_diversity_review import check
from experiments.bilingual.analyze_intent_reviews import analyze


class IntentDiversityReviewTest(unittest.TestCase):
    def test_blind_content_and_attribution_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);baseline=root/'blind-review.jsonl';review=root/'review.jsonl'
            case=dict(case_id='c1',original_need='source',request_a_zh='a',request_b_zh='b',request_a_en='a',request_b_en='b',chinese_relation=None,english_relation=None,rationale=None,reviewer_id=None)
            baseline.write_text(json.dumps(case)+'\n');(root/'manifest.json').write_text(json.dumps(dict(cases=1,output_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())))
            review.write_text(json.dumps(case)+'\n');self.assertEqual(check(root,review)['chinese_relation']['missing'],1)
            case['chinese_relation']='same';review.write_text(json.dumps(case)+'\n')
            with self.assertRaises(ValueError):check(root,review)
            case.update(reviewer_id='reviewer',rationale='evidence');review.write_text(json.dumps(case)+'\n')
            self.assertEqual(check(root,review)['chinese_relation']['same'],1)
            case['request_a_zh']='changed';review.write_text(json.dumps(case)+'\n')
            with self.assertRaises(ValueError):check(root,review)


    def test_two_reviewers_preserve_missing_and_uncertain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);baseline=root/'blind-review.jsonl';first=root/'first.jsonl';second=root/'second.jsonl'
            case=dict(case_id='c1',original_need='source',request_a_zh='a',request_b_zh='b',request_a_en='a',request_b_en='b',chinese_relation=None,english_relation=None,rationale=None,reviewer_id=None)
            baseline.write_text(json.dumps(case)+'\n');(root/'manifest.json').write_text(json.dumps(dict(cases=1,output_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())))
            first_case=dict(case,chinese_relation='same',english_relation='uncertain',rationale='reason',reviewer_id='one')
            second_case=dict(case,chinese_relation='different',english_relation='same',rationale='reason',reviewer_id='two')
            first.write_text(json.dumps(first_case)+'\n');second.write_text(json.dumps(second_case)+'\n')
            result=analyze(root,first,second)
            self.assertEqual(result['fields']['chinese_relation']['agreement'],0)
            self.assertIsNone(result['fields']['english_relation']['agreement'])
            second_case['reviewer_id']='one';second.write_text(json.dumps(second_case)+'\n')
            with self.assertRaises(ValueError):analyze(root,first,second)


if __name__=='__main__':unittest.main()
