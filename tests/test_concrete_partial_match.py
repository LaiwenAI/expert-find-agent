import json
from types import SimpleNamespace
import unittest
from expert_search.concrete_partial_match import ConcreteFocusedPartialMatcher,CONCRETE_INSTRUCTION


class ConcreteMatcherTest(unittest.TestCase):
    def test_one_call_keeps_raw_view_and_ownership_guards(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            value=dict(experts=[dict(id='c0',l='t',d=['d0'],e=['c1:s0:p0'],m='借用他人证据。',a=['需要验证。'],g=['其他模块未知。'])],read_sources=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))],model_dump=lambda **kw:{'choices':[]})
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        candidates={e:{e+'s':dict(id=e+'s',expert_id=e,kind='paper',text='Own documented method.')} for e in ['a','b']}
        result=ConcreteFocusedPartialMatcher(client).match('需求原文。',candidates,k=1)
        self.assertEqual(len(calls),1)
        self.assertIn(CONCRETE_INSTRUCTION,calls[0]['messages'][0]['content'])
        payload=json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(len(payload['candidates']),2)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['error_type'],'ValueError')
