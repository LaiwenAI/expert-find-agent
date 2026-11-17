import unittest
from expert_search.choice_anchored_candidate_agent import ChoiceAnchoredCandidateAgent


class Pipeline:
    def search_reranked(self,*args,**kwargs):
        return {'results':[{'expert':{'id':'a'},'evidence':[{'id':'s','expert_id':'a','text':'source method'}]},
                           {'expert':{'id':'b'},'evidence':[{'id':'t','expert_id':'b','text':'other source'}]}],
                'timing':{},'source_snapshot_sha256':'snapshot'}


class ChoiceAnchoredTest(unittest.TestCase):
    def test_one_call_owns_exact_spans_and_ranks(self):
        calls=[]
        def decide(client,records,questions):
            calls.append((records,questions))
            return {'labels':{'c0_level':'topic_only','c0_demand':'d0','c0_source':'c0s0',
                              'c1_level':'transferable_method','c1_demand':'d0','c1_source':'c1s0'},'seconds':.1}
        agent=ChoiceAnchoredCandidateAgent(Pipeline(),client=object(),decision_fn=decide)
        result=agent.run('demand',k=2,candidates=2,rerank_pool=2)
        self.assertEqual(len(calls),1);self.assertEqual([x['expert_id'] for x in result['results']],['b','a'])
        self.assertEqual(result['results'][0]['source_anchor']['quote'],'other source')
        self.assertEqual(result['results'][0]['demand_anchor']['quote'],'demand')
        self.assertIsNone(result['results'][0]['bridge']);self.assertEqual(result['status'],'complete')

    def test_text_descriptions_include_only_the_selected_owner_view(self):
        def decide(client,records,questions):
            source=next(q for q in questions if q['name']=='c0_source')
            self.assertIn('source method',source['choices'][1]['description'])
            self.assertNotIn('other source',source['choices'][1]['description'])
            return {'labels':{'c0_level':'topic_only','c0_demand':'d0','c0_source':'c0s0',
                              'c1_level':'insufficient','c1_demand':'none','c1_source':'none'}}
        agent=ChoiceAnchoredCandidateAgent(Pipeline(),client=object(),decision_fn=decide,text_choice_descriptions=True)
        self.assertEqual(agent.run('demand',k=1,candidates=2,rerank_pool=2)['choice_descriptions'],'text')

    def test_missing_anchor_and_api_failure_are_unresolved(self):
        def choice(*args):return {'labels':{'c0_level':'transferable_method','c0_demand':'none','c0_source':'c0s0',
                                           'c1_level':'insufficient','c1_demand':'none','c1_source':'none'}}
        result=ChoiceAnchoredCandidateAgent(Pipeline(),client=object(),decision_fn=choice).run('demand',k=2,candidates=2,rerank_pool=2)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['unresolved'],1);self.assertEqual(result['results'][0]['level'],'insufficient')
        fail=lambda *args:(_ for _ in ()).throw(TimeoutError('private text'))
        failed=ChoiceAnchoredCandidateAgent(Pipeline(),client=object(),decision_fn=fail).run('demand',k=1,candidates=2,rerank_pool=2)
        self.assertEqual(failed['status'],'failed');self.assertNotIn('private text',repr(failed))

    def test_empty_source_never_calls_model(self):
        class Empty(Pipeline):
            def search_reranked(self,*args,**kwargs):
                result=super().search_reranked(*args,**kwargs)
                for row in result['results']:row['evidence']=[]
                return result
        def fail(*args):raise AssertionError('Model should not be called')
        result=ChoiceAnchoredCandidateAgent(Empty(),client=object(),decision_fn=fail).run('demand',k=1,candidates=2,rerank_pool=2)
        self.assertEqual(result['status'],'unavailable');self.assertEqual(result['results'],[])

    def test_foreign_choice_is_rejected(self):
        def choice(*args):return {'labels':{'c0_level':'topic_only','c0_demand':'d0','c0_source':'c1s0',
                                           'c1_level':'topic_only','c1_demand':'d0','c1_source':'c1s0'}}
        result=ChoiceAnchoredCandidateAgent(Pipeline(),client=object(),decision_fn=choice).run('demand',k=1,candidates=2,rerank_pool=2)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['results'],[])


if __name__=='__main__':unittest.main()
