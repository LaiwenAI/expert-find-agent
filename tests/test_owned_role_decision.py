import unittest
from types import SimpleNamespace

from expert_search.owned_role_decision import compile_owned_decision_roles
from expert_search.owned_role_decision_order import decide_roles


class Client:
    def __init__(self):self.decisions=SimpleNamespace(create=self.create)
    def create(self,**payload):
        self.payload=payload;answers=[]
        for q in payload['questions']:
            answers.append({'type':'choice','name':q['name'],'choice':'uncertain',
                            'probabilities':[{'value':c['value'],'probability':1.0 if c['value']=='uncertain' else 0.0} for c in q['choices']]})
        return SimpleNamespace(model_dump=lambda **kwargs:{'model':'gpt-6-luna','answers':answers,'usage':{'input_tokens':10,'output_tokens':2}})


class OwnedRoleDecisionTest(unittest.TestCase):
    def test_native_choices_are_unique_and_bound_to_source(self):
        client=Client();source={'id':'p','expert_id':'e','text':'abcdefgh'}
        result=compile_owned_decision_roles(source,client,width=4)
        self.assertEqual(result['status'],'normalized_role_hints');self.assertTrue(result['usage_present'])
        self.assertEqual(result['role_record']['roles']['uncertain'],[0,1])
        self.assertEqual(result['role_record']['normalization']['conflicts'],0)
        self.assertEqual(len(client.payload['questions']),2)

    def test_bad_client_falls_back(self):
        result=compile_owned_decision_roles({'id':'p','expert_id':'e','text':'text'},object())
        self.assertEqual(result['status'],'raw_fallback');self.assertFalse(result['usage_present'])

    def test_order_control_changes_only_choice_order(self):
        source={'id':'p','expert_id':'e','text':'abcdefgh'};a=Client();b=Client()
        self.assertEqual(decide_roles(source,a,'forward',width=4)['status'],'normalized_role_hints')
        self.assertEqual(decide_roles(source,b,'reverse',width=4)['status'],'normalized_role_hints')
        self.assertEqual(a.payload['input'],b.payload['input'])
        for first,second in zip(a.payload['questions'],b.payload['questions']):
            self.assertEqual(first['name'],second['name']);self.assertEqual(first['instructions'],second['instructions'])
            self.assertEqual(first['choices'],list(reversed(second['choices'])))


if __name__=='__main__':unittest.main()
