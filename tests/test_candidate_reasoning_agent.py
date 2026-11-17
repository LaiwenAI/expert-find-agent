import unittest

from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


class FakePipeline:
    def search(self,query,k,evidence_per_expert):
        return {'results':[{'expert':{'id':'a'},'evidence':[{'id':'s1','expert_id':'a','text':'a'*7000}]},
                           {'expert':{'id':'b'},'evidence':[{'id':'s2','expert_id':'b','text':'source b'}]}],
                'timing':{'total_seconds':0}}


class CandidateReasoningAgentTest(unittest.TestCase):
    def test_rejects_cross_owner_citations_and_truncates_visible_text(self):
        agent=CandidateReasoningAgent(FakePipeline(),client=object());captured={}
        def ask(stage,instruction,payload):
            captured.update(payload)
            return {'results':[{'expert_id':'a','citations':['s2']},{'expert_id':'b','citations':['s2']},
                               {'expert_id':'a','citations':['s1']}]}
        agent.helper.ask=ask
        result=agent.run('need',k=2,candidates=2)
        self.assertEqual([r['expert']['id'] for r in result['results']],['b','a'])
        self.assertEqual(len(captured['evidence'][0]['text']),6000)
        self.assertEqual(len(result['results'][1]['evidence'][0]['text']),6000)
        with self.assertRaises(ValueError):agent.run('need',k=2,candidates=2,evidence_per_expert=0)


if __name__=='__main__':unittest.main()
