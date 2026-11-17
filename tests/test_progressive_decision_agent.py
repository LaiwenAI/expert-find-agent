import unittest
from unittest.mock import patch

from expert_search.agent import ExpertAgent
from expert_search.progressive_decision_agent import ProgressiveDecisionAgent


class Pipeline:
    def search_reranked(self,query,k,candidates,evidence_per_expert):
        return {'results':[{'expert':{'id':'a'},'score':1.,'evidence':[{'id':'s1','expert_id':'a','text':'one'}]},
                           {'expert':{'id':'b'},'score':.5,'evidence':[{'id':'s2','expert_id':'b','text':'two'}]}],
                'timing':{'total_seconds':0.},'source_snapshot_sha256':'fixture'}


class ProgressiveDecisionTest(unittest.TestCase):
    def test_screen_then_explain_is_lazy_and_validates_owned_quote(self):
        calls=[]
        def choice(client,records,questions):
            calls.append('screen');return {'labels':{'c0':'topic_only','c1':'transferable_method'}}
        agent=ProgressiveDecisionAgent(Pipeline(),client=object(),decision_fn=choice)
        answer={'results':[{'expert_id':'b','reason':'reason','missing':[], 'citations':[{'evidence_id':'s2','quote':'two'}]}]}
        with patch.object(ExpertAgent,'ask',return_value=answer) as ask:
            events=agent.events('query',k=1,candidates=2,rerank_pool=2,explanation_candidates=1)
            first=next(events);self.assertEqual(first['stage'],'retrieval');self.assertFalse(calls);ask.assert_not_called()
            first['results'][0]['evidence'][0]['text']='altered'
            screen=next(events);self.assertEqual(screen['stage'],'screen');self.assertEqual(screen['results'][0]['expert']['id'],'b');ask.assert_not_called()
            screen['results'][0]['evidence'][0]['text']='altered'
            final=next(events);self.assertEqual(final['stage'],'explanation');self.assertEqual(final['results'][0]['evidence'][0]['text'],'two')
            self.assertEqual(final['completion']['status'],'complete')

    def test_screen_shortage_is_partial_against_original_budget(self):
        def choice(*_):return {'labels':{'c0':'topic_only','c1':'transferable_method'}}
        agent=ProgressiveDecisionAgent(Pipeline(),client=object(),decision_fn=choice)
        answer={'results':[{'expert_id':'b','reason':'reason','missing':[], 'citations':[{'evidence_id':'s2','quote':'two'}]}]}
        with patch.object(ExpertAgent,'ask',return_value=answer):
            events=list(agent.events('query',k=2,candidates=2,rerank_pool=2,explanation_candidates=2))
        self.assertEqual(events[-1]['completion']['requested'],2)
        self.assertEqual(events[-1]['completion']['status'],'partial')
        self.assertEqual(events[-1]['explanation_call_completion']['status'],'complete')

    def test_empty_source_is_not_screened(self):
        class EmptyPipeline:
            def search_reranked(self,query,k,candidates,evidence_per_expert):
                return {'results':[{'expert':{'id':'a'},'score':0.,'evidence':[{'id':'s','expert_id':'a','text':' '}]}],
                        'timing':{'total_seconds':0.},'source_snapshot_sha256':'fixture'}
        def fail_if_called(*_):raise AssertionError('Decision must not run on empty evidence')
        agent=ProgressiveDecisionAgent(EmptyPipeline(),client=object(),decision_fn=fail_if_called)
        events=list(agent.events('query',k=1,candidates=1,rerank_pool=1,explanation_candidates=1))
        self.assertEqual([event['stage'] for event in events],['retrieval','screen_unavailable'])

    def test_abstention_and_screen_failure_do_not_generate_explanation(self):
        for choice, expected in ((lambda *_:{'labels':{'c0':'insufficient','c1':'topic_only'}}, 'screen'),
                                 (lambda *_:(_ for _ in ()).throw(TimeoutError()), 'screen_failed')):
            agent=ProgressiveDecisionAgent(Pipeline(),client=object(),decision_fn=choice)
            with patch.object(ExpertAgent,'ask') as ask:
                events=list(agent.events('query',k=1,candidates=2,rerank_pool=2,explanation_candidates=1))
            self.assertEqual(events[0]['stage'],'retrieval');self.assertEqual(events[1]['stage'],expected);ask.assert_not_called()
            if expected=='screen':self.assertEqual(events[2]['reason'],'no_positive_screen')


if __name__=='__main__':unittest.main()
