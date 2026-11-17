import unittest
from expert_search.lexical_anchored_choice_agent import LexicalAnchoredChoiceAgent,propose_pair


class Pipeline:
    def search_reranked(self,*args,**kwargs):
        return {'results':[{'expert':{'id':'a'},'evidence':[{'id':'s','expert_id':'a','text':'sensor calibration method'}]},
                           {'expert':{'id':'b'},'evidence':[{'id':'t','expert_id':'b','text':'quantum optics'}]}],
                'timing':{},'source_snapshot_sha256':'snapshot'}


class LexicalAnchoredTest(unittest.TestCase):
    def test_zero_overlap_is_unknown_and_source_bound(self):
        calls=[]
        def choice(client,records,questions):
            calls.append(questions);self.assertEqual([q['name'] for q in questions],['c0'])
            return {'labels':{'c0':'transferable_method'},'seconds':.1}
        result=LexicalAnchoredChoiceAgent(Pipeline(),client=object(),decision_fn=choice).run('sensor calibration demand',k=2,candidates=2,rerank_pool=2)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['unresolved'],1)
        self.assertEqual(result['results'][0]['source_anchor']['evidence_id'],'s');self.assertIsNone(result['results'][0]['bridge']);self.assertEqual(result['model_positive_returned'],1)
        self.assertEqual(result['results'][0]['source_anchor']['quote'],'sensor calibration method')
        self.assertEqual(len(calls),1)

    def test_deterministic_selection_and_stopwords(self):
        sources=[{'id':'s','expert_id':'a','text':'sensor calibration. other sensor work.'}]
        self.assertEqual(propose_pair('the and for',sources),None)
        first=propose_pair('sensor calibration',sources,width=12);self.assertEqual(first,propose_pair('sensor calibration',sources,width=12))
        self.assertEqual(first['source_anchor']['expert_id'],'a')

    def test_invalid_decision_does_not_fabricate_results(self):
        bad=lambda *args:{'labels':{'foreign':'direct_application'}}
        result=LexicalAnchoredChoiceAgent(Pipeline(),client=object(),decision_fn=bad).run('sensor calibration',k=1,candidates=2,rerank_pool=2)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['results'],[])


if __name__=='__main__':unittest.main()
