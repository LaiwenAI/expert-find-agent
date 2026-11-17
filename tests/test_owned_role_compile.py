import json
import unittest
from types import SimpleNamespace

from expert_search.owned_role_compile import compile_owned_roles


class Client:
    def __init__(self,content):
        self.content=content;self.chat=SimpleNamespace(completions=self)

    def create(self,**kwargs):
        self.messages=kwargs['messages']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5),model='test-model')


class OwnedRoleCompileTest(unittest.TestCase):
    def test_query_blind_normalization_and_failure_fallback(self):
        source={'id':'p','expert_id':'e','text':'abcdefgh'}
        roles={'method':[0],'result':[],'interpretation':[],'background':[],'other':[],'uncertain':[]}
        client=Client(json.dumps(roles));result=compile_owned_roles(source,client,width=4)
        self.assertEqual(result['status'],'normalized_role_hints')
        self.assertEqual(result['role_record']['roles']['uncertain'],[1])
        self.assertTrue(result['usage_present'])
        self.assertEqual(set(json.loads(client.messages[1]['content'])),{'windows'})
        failed=compile_owned_roles(source,Client('not json'),width=4)
        self.assertEqual(failed['status'],'raw_fallback');self.assertIsNone(failed['role_record'])
        self.assertFalse(failed['usage_present'])
        with self.assertRaises(ValueError):compile_owned_roles(source,client,width=1,max_windows=2)


if __name__=='__main__':unittest.main()
