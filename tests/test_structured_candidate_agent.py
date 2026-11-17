import unittest
from expert_search.structured_candidate_agent import StructuredCandidateAgent


class Pipeline:
    def search_reranked(self,*args,**kwargs):
        return {'results':[{'expert':{'id':'a'},'evidence':[{'id':'s','expert_id':'a','kind':'paper','text':'source method'}]}],
                'timing':{},'source_snapshot_sha256':'snapshot'}


class Matcher:
    def __init__(self,quote='source'):self.quote=quote
    def match(self,demand,candidates,cards,mode):
        return {'status':'structurally_valid','result':{'ranking':[{'expert_id':'a','level':'transferable_method',
               'demand_citations':[{'start':0,'end':6,'quote':'demand'}],
               'evidence':[{'expert_id':'a','citations':[{'evidence_id':'s','start':0,'end':6,'quote':self.quote}]}],
               'mechanism':'connection','assumptions':['validate'],'remaining_demand_gaps':['other']} ]}}


class StructuredAgentTest(unittest.TestCase):
    def test_validates_exact_demand_and_owned_source(self):
        agent=StructuredCandidateAgent(Pipeline(),matcher=Matcher())
        result=agent.run('demand',k=1,candidates=1,rerank_pool=1)
        self.assertEqual(result['results'][0]['expert']['id'],'a');self.assertEqual(result['status'],'structurally_valid')
        with self.assertRaisesRegex(ValueError,'source span'):StructuredCandidateAgent(Pipeline(),matcher=Matcher('invent')).run('demand',k=1,candidates=1,rerank_pool=1)

    def test_missing_and_failures_do_not_invent_answers(self):
        class Empty(Pipeline):
            def search_reranked(self,*a,**k):
                return {'results':[],'timing':{},'source_snapshot_sha256':'snapshot'}
        self.assertEqual(StructuredCandidateAgent(Empty(),matcher=Matcher()).run('demand',k=1,candidates=1,rerank_pool=1)['status'],'unavailable')
        class Failure:
            def match(self,*args):return {'status':'unresolved'}
        self.assertEqual(StructuredCandidateAgent(Pipeline(),matcher=Failure()).run('demand',k=1,candidates=1,rerank_pool=1)['results'],[])


if __name__=='__main__':unittest.main()
