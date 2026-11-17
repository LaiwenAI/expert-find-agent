import hashlib
import unittest
from expert_search.method_frame_store import bind_frames,as_fact_card


class MethodFrameStoreTest(unittest.TestCase):
    def setUp(self):
        self.source=dict(id='s',expert_id='e',kind='paper',text='Own documented operation.')
        self.record=dict(source_id='s',expert_id='e',source_sha256=hashlib.sha256(self.source['text'].encode()).hexdigest(),frames=[dict(id='f0',source_id='s',expert_id='e',operation='Documented operation.',input=None,output='Output.',conditions=['Only under stated condition.'],citations=[dict(evidence_id='s',start=0,end=25,quote='Own documented operation.')])],audit=dict(labels=dict(f0='supported_frame')))

    def test_source_binding_and_nullable_fields_survive_adapter(self):
        bound=bind_frames(self.source,self.record);card=as_fact_card(bound)
        self.assertTrue(bound['raw_source_required'])
        self.assertIn('"input": null',card['cards'][0]['claim'])
        self.assertIn('Only under stated condition.',card['cards'][0]['claim'])
        empty=dict(self.record,frames=[]);self.assertTrue(bind_frames(self.source,empty)['raw_source_required'])

    def test_stale_source_and_borrowed_quote_rejected(self):
        with self.assertRaises(ValueError):bind_frames(dict(self.source,text='Changed source.'),self.record)
        self.record['frames'][0]['citations'][0]['evidence_id']='foreign'
        with self.assertRaises(ValueError):bind_frames(self.source,self.record)
