import unittest
from expert_search.typed_node_router import TypedNodeRouter,LEVELS


class Client:
    def __init__(self,answers):self.answers=answers
    def post(self,path,**kwargs):return dict(answers=self.answers,model='gpt-6-luna')


def answer(name,label='transferable_method'):
    return dict(name=name,type='choice',choice=label,probabilities=[dict(value=value,probability=float(value==label)) for value in LEVELS])


class TypedNodeRouterTest(unittest.TestCase):
    def test_unknown_node_response_cannot_be_attached_to_valid_expert(self):
        with self.assertRaisesRegex(ValueError,'router answers'):
            TypedNodeRouter(Client([answer('other')])).route('Find a relevant research method.',[dict(id='n0',description='Documented robot planning.')])

    def test_refusal_or_fact_label_cannot_become_applicability(self):
        for response in (dict(name='n0',type='refusal'),answer('n0','supported')):
            with self.subTest(response=response),self.assertRaisesRegex(ValueError,'routing choice'):
                TypedNodeRouter(Client([response])).route('Find a relevant research method.',[dict(id='n0',description='Documented robot planning.')])
