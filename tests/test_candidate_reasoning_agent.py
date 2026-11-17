import unittest

from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


class FakePipeline:
    def search(self,query,k,evidence_per_expert):
        return {'results':[{'expert':{'id':'a'},'evidence':[{'id':'s1','expert_id':'a','text':'a'*7000}]},
                           {'expert':{'id':'b'},'evidence':[{'id':'s2','expert_id':'b','text':'source b'}]}],
                'timing':{'total_seconds':0}}

    def search_reranked(self,query,k,candidates,evidence_per_expert):
        self.rerank_args=(query,k,candidates,evidence_per_expert)
        return self.search(query,k,evidence_per_expert)


class CandidateReasoningAgentTest(unittest.TestCase):
    def test_rejects_cross_owner_citations_and_truncates_visible_text(self):
        agent=CandidateReasoningAgent(FakePipeline(),client=object());captured={}
        def ask(stage,instruction,payload):
            captured.update(payload)
            return {'results':[{'expert_id':'a','reason':'x','missing':[],'citations':[{'evidence_id':'s2','quote':'source'}]},
                               {'expert_id':'b','reason':'y','missing':[],'citations':[{'evidence_id':'s2','quote':'source b'}]},
                               {'expert_id':'a','reason':'z','missing':[],'citations':[{'evidence_id':'s1','quote':'aaaa'}]}]}
        agent.helper.ask=ask
        result=agent.run('need',k=2,candidates=2)
        self.assertEqual([r['expert']['id'] for r in result['results']],['b','a'])
        self.assertEqual(len(captured['evidence'][0]['text']),6000)
        self.assertEqual(captured['evidence'][0]['contribution_status'], 'unverified')
        self.assertEqual(captured['evidence'][0]['authorship_association'], 'source_association_not_independently_verified')
        self.assertIsNone(captured['evidence'][0]['publication'])
        self.assertEqual(len(result['results'][1]['evidence'][0]['text']),6000)
        self.assertEqual(result['results'][0]['evidence'][0]['start'],0)
        self.assertEqual(result['results'][0]['evidence'][0]['end'],8)
        self.assertEqual(result['rejected_results'],1)
        self.assertEqual(result['completion']['status'],'complete')
        with self.assertRaises(ValueError):agent.run('need',k=2,candidates=2,evidence_per_expert=0)
        result=agent.run('need',k=2,candidates=2,retrieval='adaptive_bm25',rerank_pool=4)
        self.assertEqual(agent.pipeline.rerank_args,('need',2,4,1))
        self.assertEqual(result['retrieval_strategy'],'adaptive_bm25')
        with self.assertRaises(ValueError):agent.run('need',k=2,candidates=2,retrieval='adaptive_bm25',rerank_pool=1)
        with self.assertRaises(ValueError):agent.run('need',k=2,candidates=2,retrieval='invalid')

    def test_paper_citation_context_is_scoped_and_unverified(self):
        class Pipeline(FakePipeline):
            def search(self,*args,**kwargs):
                data=super().search(*args,**kwargs);data['results'][0]['evidence'][0].update(kind='paper',citation_context={
                    'observed_total':12,'observation_date':None,'total_history_status':'partial_or_unresolved_history',
                    'field_normalized_impact':None,'personal_credit':None})
                return data
        agent=CandidateReasoningAgent(Pipeline(),client=object());captured={}
        def ask(stage,instruction,payload):
            captured['instruction']=instruction;captured['payload']=payload
            return {'results':[{'expert_id':'a','reason':'r','missing':[],'citations':[{'evidence_id':'s1','quote':'aaaa'}]}]}
        agent.helper.ask=ask;result=agent.run('need',k=1,candidates=2)
        self.assertIn('never infer individual credit',captured['instruction'])
        self.assertEqual(captured['payload']['evidence'][0]['citation_context']['observed_total'],12)
        self.assertIsNone(result['results'][0]['evidence'][0]['citation_context']['personal_credit'])

    def test_citation_context_cannot_claim_personal_credit(self):
        class Pipeline(FakePipeline):
            def search(self,*args,**kwargs):
                data=super().search(*args,**kwargs);data['results'][0]['evidence'][0].update(kind='paper',citation_context={
                    'observed_total':12,'observation_date':None,'total_history_status':'partial_or_unresolved_history',
                    'field_normalized_impact':None,'personal_credit':12})
                return data
        agent=CandidateReasoningAgent(Pipeline(),client=object())
        with self.assertRaisesRegex(ValueError,'citation context'):agent.run('need',k=1,candidates=2)

    def test_reports_partial_results_without_fabricating_missing_experts(self):
        agent=CandidateReasoningAgent(FakePipeline(),client=object())
        agent.helper.ask=lambda *_: {'results':[{'expert_id':'b','reason':'r','missing':[],'citations':[{'evidence_id':'s2','quote':'source b'}]}]}
        result=agent.run('need',k=2,candidates=2)
        self.assertEqual(result['completion']['status'],'partial')
        self.assertEqual(result['completion']['returned'],1)
        self.assertEqual([row['expert']['id'] for row in result['results']],['b'])

    def test_rejects_invented_and_truncated_quotes(self):
        agent=CandidateReasoningAgent(FakePipeline(),client=object())
        agent.helper.ask=lambda *_: {'results':[{'expert_id':'b','reason':'r','missing':[],'citations':[{'evidence_id':'s2','quote':'invented'}]},
                                               {'expert_id':'a','reason':'r','missing':[],'citations':[{'evidence_id':'s1','quote':'a'*6001}]}]}
        result=agent.run('need',k=2,candidates=2)
        self.assertEqual(result['results'],[])
        self.assertEqual(result['rejected_results'],2)
        self.assertEqual(result['completion']['status'],'empty')
        self.assertEqual(result['completion']['requested'],2)
        self.assertEqual(result['completion']['returned'],0)


if __name__=='__main__':unittest.main()
