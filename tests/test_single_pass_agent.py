import unittest
from expert_search.single_pass_agent import SinglePassAgent


class SearchStub:
    experts={'a':{'id':'a'}, 'b':{'id':'b'}}
    def search(self, query, candidates, exclude=()):
        return [{'id':'a','evidence':[{'id':'ea','expert_id':'a','text':'evidence'}]},
                {'id':'b','evidence':[{'id':'eb','expert_id':'b','text':'evidence'}]}]


class SinglePassTest(unittest.TestCase):
    def test_rejects_foreign_evidence_and_keeps_owned_citation(self):
        agent=SinglePassAgent.__new__(SinglePassAgent);agent.searcher=SearchStub();agent.usage={'prompt_tokens':0}
        agent.ask=lambda *args: {'results':[{'expert_id':'a','citations':['eb']},
                                             {'expert_id':'b','citations':['missing','eb','eb']}]}
        result=agent.run('need',k=2,candidates=2)
        self.assertEqual([x['expert']['id'] for x in result['results']],['b'])
        self.assertEqual([x['id'] for x in result['results'][0]['evidence']],['eb'])
        self.assertEqual(result['candidate_ids'],['a','b'])
