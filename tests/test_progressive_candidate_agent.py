import unittest
from unittest.mock import patch

from expert_search.agent import ExpertAgent
from expert_search.progressive_candidate_agent import ProgressiveCandidateAgent


class Pipeline:
    def search_reranked(self, query, k, candidates, evidence_per_expert):
        return {'results':[{'expert':{'id':'a'},'score':1.,'evidence':[{'id':'s','expert_id':'a','text':'document'}]}],
                'timing':{'total_seconds':0.},'source_snapshot_sha256':'fixture'}


class ProgressiveTest(unittest.TestCase):
    def test_first_event_precedes_model_and_is_isolated(self):
        agent=ProgressiveCandidateAgent(Pipeline(),client=object())
        answer={'results':[{'expert_id':'a','reason':'reason','missing':[],
                            'citations':[{'evidence_id':'s','quote':'document'}]}]}
        with patch.object(ExpertAgent,'ask',return_value=answer) as ask:
            events=agent.events('query',k=1,candidates=1,rerank_pool=1)
            first=next(events); self.assertEqual(first['stage'],'retrieval'); ask.assert_not_called()
            first['results'][0]['evidence'][0]['text']='modified by consumer'
            second=next(events);self.assertEqual(second['stage'],'explanation')
            self.assertEqual(second['completion']['status'],'complete')
            self.assertEqual(second['results'][0]['evidence'][0]['text'],'document')
            self.assertEqual(ask.call_count,1)
            with self.assertRaises(StopIteration): next(events)

    def test_empty_evidence_does_not_call_model(self):
        class EmptyPipeline:
            def search_reranked(self, query, k, candidates, evidence_per_expert):
                return {'results': [], 'timing': {'total_seconds': 0.}, 'source_snapshot_sha256': 'fixture'}
        agent=ProgressiveCandidateAgent(EmptyPipeline(),client=object())
        with patch.object(ExpertAgent,'ask') as ask:
            events=list(agent.events('query',k=1,candidates=1,rerank_pool=1))
        self.assertEqual([event['stage'] for event in events],['retrieval','explanation_unavailable'])
        ask.assert_not_called()

    def test_model_failure_does_not_certify_retrieval(self):
        agent=ProgressiveCandidateAgent(Pipeline(),client=object())
        with patch.object(ExpertAgent,'ask',side_effect=TimeoutError):
            events=list(agent.events('query',k=1,candidates=1,rerank_pool=1))
        self.assertEqual([event['stage'] for event in events],['retrieval','explanation_failed'])
        self.assertEqual(events[1]['error_type'],'TimeoutError')
        self.assertNotIn('results',events[1])


if __name__=='__main__':unittest.main()
