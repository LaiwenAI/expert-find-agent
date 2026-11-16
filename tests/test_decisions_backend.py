import unittest
from expert_search.decisions_backend import DecisionsBackend


class Transport:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def post(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return {'answers': self.answers}


def answer(name, label='unknown'):
    return dict(name=name, type='choice', choice=label, probabilities=[
        dict(value=v, probability=float(v == label))
        for v in ('supported', 'contradicted', 'unknown')])


class DecisionsTest(unittest.TestCase):
    def test_typed_sdk_resource_and_error_without_transport_fallback(self):
        class Response:
            def model_dump(self,mode):
                return dict(answers=[answer('a','supported')],model='gpt-6-luna')
        class Resource:
            def create(self,**kwargs):
                self.payload=kwargs
                return Response()
        class Client:
            decisions=Resource()
            def post(self,*args,**kwargs):raise AssertionError('Unexpected raw fallback')
        client=Client()
        pairs=[dict(id='a',constraint='x',text='y')]
        result=DecisionsBackend(client).judge(pairs)
        self.assertEqual(result['transport'],'sdk-decisions-create')
        self.assertEqual(client.decisions.payload['model'],'gpt-6-luna')
        class Failed:
            def create(self,**kwargs):raise RuntimeError('unsupported operation')
        client.decisions=Failed()
        with self.assertRaisesRegex(RuntimeError,'unsupported'):DecisionsBackend(client).judge(pairs)

    def test_reordered_answers_and_native_endpoint(self):
        client = Transport([answer('b'), answer('a', 'supported')])
        pairs = [dict(id=i, constraint='robot control', text='robotics') for i in ('a', 'b')]
        result = DecisionsBackend(client).judge(pairs)
        self.assertEqual([r['id'] for r in result['judgments']], ['a', 'b'])
        self.assertEqual(client.calls[0][0], '/decisions')
        self.assertEqual(client.calls[0][1]['cast_to'],dict[str,object])
        self.assertNotIn('quote', result['judgments'][0])

    def test_duplicate_refusal_and_invalid_probability_rejected(self):
        pairs = [dict(id=i, constraint='x', text='y') for i in ('a', 'b')]
        cases = [[answer('a'), answer('a')],
                 [dict(name='a', type='refusal'), answer('b')]]
        bad = answer('a'); bad['probabilities'][0]['probability'] = float('nan')
        cases.append([bad, answer('b')])
        for answers in cases:
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                DecisionsBackend(Transport(answers)).judge(pairs)

    def test_transport_error_propagates(self):
        class Failed:
            def post(self, *args, **kwargs):
                raise RuntimeError('unsupported operation')
        with self.assertRaisesRegex(RuntimeError, 'unsupported'):
            DecisionsBackend(Failed()).judge([dict(id='a', constraint='x', text='y')])
