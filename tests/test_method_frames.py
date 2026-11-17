import json
from types import SimpleNamespace
import unittest
from expert_search.method_frames import compile_method_frames,resolve
from expert_search.span_applicability import source_spans


class MethodFrameTest(unittest.TestCase):
    def test_concrete_profile_operation_allowed_and_missing_io_preserved(self):
        source=dict(id='profile',expert_id='e',kind='profile',text='\u672c\u9879\u76ee\u7814\u53d1\u4e86\u4e00\u79cd\u7528\u4e8e\u6e29\u5ea6\u9884\u6d4b\u7684\u6709\u9650\u5143\u6c42\u89e3\u65b9\u6cd5\u3002')
        value=dict(frames=[dict(operation='\u91c7\u7528\u6709\u9650\u5143\u6c42\u89e3\u8fdb\u884c\u6e29\u5ea6\u9884\u6d4b\u3002',input=None,output='\u6e29\u5ea6\u9884\u6d4b\u7ed3\u679c\u3002',conditions=[],span_ids=['p0'])])
        calls=[]
        def chat(**kw):
            calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))],model_dump=lambda **kwargs:{})
        def decisions(**kw):
            answers=[dict(name=q['name'],type='choice',choice='supported_frame',probabilities=[dict(value=c['value'],probability=1. if c['value']=='supported_frame' else 0.) for c in q['choices']]) for q in kw['questions']]
            return SimpleNamespace(model_dump=lambda **kwargs:dict(answers=answers))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),decisions=SimpleNamespace(create=decisions))
        r=compile_method_frames(source,client)
        self.assertEqual(r['status'],'model_screened_frames')
        self.assertIsNone(r['frames'][0]['input'])
        self.assertEqual(r['frames'][0]['expert_id'],'e')
        self.assertEqual(r['api_attempts'],2)
        payload=json.loads(calls[0]['messages'][1]['content']);self.assertNotIn('target_query',payload)

    def test_foreign_span_rejected_and_empty_frames_not_inability(self):
        source=dict(id='s',expert_id='e',kind='paper',text='Documented own operation.')
        refs=source_spans('e',{'s':source})
        value=dict(frames=[dict(operation='A documented method.',input=None,output=None,conditions=[],span_ids=['foreign'])])
        with self.assertRaises(ValueError):resolve(source,refs,value)
        self.assertEqual(resolve(source,refs,dict(frames=[])),[])
