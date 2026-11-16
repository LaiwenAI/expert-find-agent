import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from expert_search.scoped_silver_benchmark import evaluate,literal_topics,load


class SilverBenchmarkTest(unittest.TestCase):
    def cases(self):
        return [dict(pair_id='p',variant='base',label='supported',input=dict(conditions=['alpha','beta'],evidence=[dict(text='Alpha'),dict(text='Beta')])),
                dict(pair_id='p',variant='counterfactual',label='unknown',input=dict(conditions=['alpha','gamma'],evidence=[dict(text='Alpha'),dict(text='Beta')]))]

    def test_labels_not_exposed_and_two_variants_scored(self):
        cases=self.cases()
        def predict(payload):
            self.assertNotIn('label',payload)
            return literal_topics(payload)
        result=evaluate(cases,predict)
        self.assertEqual(result['silver_state_accuracy'],1)
        self.assertEqual(result['both_variants_correct'],1)

    def test_abstention_not_reported_as_perfect_precision(self):
        result=evaluate(self.cases(),lambda _: 'unknown')
        self.assertIsNone(result['supported_precision'])
        self.assertEqual(result['both_variants_correct'],0)
        with self.assertRaises(ValueError):evaluate(self.cases()[:1],lambda _: 'unknown')

    def test_full_input_snapshot_owner_and_content_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            text='alpha beta'
            snapshot=dict(evidence_id='s',expert_id='e',source='file:1',prefix_characters=len(text),
                          source_snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())
            pair=dict(id='p',constructing_expert_id='e',query='base',counterfactual_query='changed',
                      source_snapshots=[snapshot],constraints=['alpha','beta'],changed_index=1,
                      replacement_text='gamma',base_source_support='supported',counterfactual_source_support='unknown')
            (root/'pairs.jsonl').write_text(json.dumps(pair)+'\n')
            source=dict(id='s',expert_id='e',source='file:1',text=text+' extra context not supplied')
            (root/'evidence.jsonl').write_text(json.dumps(source)+'\n')
            self.assertEqual(load(root,'all',root)[0]['input']['evidence'][0]['text'],text)
            source['expert_id']='other expert'
            (root/'evidence.jsonl').write_text(json.dumps(source)+'\n')
            with self.assertRaises(ValueError):load(root,'all',root)
            source['expert_id']='e'
            source['text']='changed prefix'
            (root/'evidence.jsonl').write_text(json.dumps(source)+'\n')
            with self.assertRaises(ValueError):load(root,'all',root)
