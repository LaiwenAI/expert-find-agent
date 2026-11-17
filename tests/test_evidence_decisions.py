import unittest
from expert_search.evidence_decisions import EvidenceDecisions


class Client:
    def __init__(self,answers):self.answers=answers
    def post(self,*args,**kwargs):return dict(answers=self.answers)


def answer(name):return dict(name=name,type='choice',choice='evidence',probabilities=[dict(value='evidence',probability=.7),dict(value='other',probability=.3)])


class EvidenceDecisionsTest(unittest.TestCase):
    def test_foreign_answer_cannot_be_attached_to_a_known_sentence(self):
        with self.assertRaisesRegex(ValueError,'foreign'):
            EvidenceDecisions(Client([answer('s99')])).score('A hypothesis',{'s0':'A sentence.'},[])

    def test_invalid_probability_or_refusal_never_becomes_relevance(self):
        bad=answer('s0');bad['probabilities'][0]['probability']=float('nan')
        for row in (bad,dict(name='s0',type='refusal')):
            with self.assertRaises(ValueError):EvidenceDecisions(Client([row])).score('A hypothesis',{'s0':'A sentence.'},[])
