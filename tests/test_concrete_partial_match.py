import json
from types import SimpleNamespace
import unittest
from expert_search.concrete_partial_match import ConcreteFocusedPartialMatcher,CONCRETE_INSTRUCTION


class ConcreteMatcherTest(unittest.TestCase):
    def test_one_call_keeps_raw_view_and_ownership_guards(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            value=dict(experts=[dict(id='c0',l='t',d=['d0'],e=['c1:s0:p0'],m='\u501f\u7528\u4ed6\u4eba\u8bc1\u636e\u3002',a=['\u9700\u8981\u9a8c\u8bc1\u3002'],g=['\u5176\u4ed6\u6a21\u5757\u672a\u77e5\u3002'])],read_sources=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))],model_dump=lambda **kw:{'choices':[]})
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        candidates={e:{e+'s':dict(id=e+'s',expert_id=e,kind='paper',text='Own documented method.')} for e in ['a','b']}
        result=ConcreteFocusedPartialMatcher(client).match('\u9700\u6c42\u539f\u6587\u3002',candidates,k=1)
        self.assertEqual(len(calls),1)
        self.assertIn(CONCRETE_INSTRUCTION,calls[0]['messages'][0]['content'])
        payload=json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(len(payload['candidates']),2)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['error_type'],'ValueError')
