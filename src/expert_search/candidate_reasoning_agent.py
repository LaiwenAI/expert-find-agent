"""One-call evidence-grounded ranking over a candidate-first retrieval pipeline."""
import time

from .agent import ExpertAgent
from .candidate_evidence_pipeline import CandidateEvidencePipeline


class CandidateReasoningAgent:
    def __init__(self,pipeline=None,client=None):
        self.pipeline=pipeline if pipeline is not None else CandidateEvidencePipeline()
        if client is None:
            from laiwenai import LaiwenAI
            client=LaiwenAI(model='gpt-6-luna')
        self.helper=ExpertAgent.__new__(ExpertAgent)
        self.helper.client=client;self.helper.trace=[];self.helper.usage={'prompt_tokens':0,'completion_tokens':0}

    def run(self,query,k=5,candidates=20,evidence_per_expert=1):
        if not isinstance(query,str) or not query.strip() or type(k)!=int or type(candidates)!=int or not 1<=k<=candidates<=50:
            raise ValueError('Require a nonempty query and 1 <= k <= candidates <= 50')
        if type(evidence_per_expert)!=int or not 1<=evidence_per_expert<=3:raise ValueError('Require 1 to 3 evidence items per expert')
        start=time.perf_counter();self.helper.trace=[];self.helper.usage=dict.fromkeys(self.helper.usage,0)
        retrieved=self.pipeline.search(query,candidates,evidence_per_expert)
        payload=[];visible={};experts={}
        for row in retrieved['results']:
            eid=row['expert']['id'];experts[eid]=row['expert'];items=[]
            for e in row['evidence']:
                if e['expert_id']!=eid or e['id'] in visible:raise ValueError('Invalid evidence ownership or duplicate evidence ID')
                item={'id':e['id'],'expert_id':eid,'text':e['text'][:6000],
                      'kind':e.get('kind','unknown'), 'publication':e.get('publication'),
                      'contribution_status':'unverified',
                      'authorship_association':'source_association_not_independently_verified'}
                visible[e['id']]=item;items.append(item)
            if items:payload.extend(items)
        if not payload:raise ValueError('No candidate evidence')
        response=self.helper.ask('verify','Rank experts using only supplied evidence. Cite evidence IDs and state missing requirements. Paper ownership is an unverified association, not proof of individual contribution or method credit. Respect publication dates when present and state when date or contribution is unknown. Do not infer capability from names or institutions. Return {"results":[{"expert_id":str,"reason":str,"missing":[str],"citations":[str]}]}.',{'demand':query,'evidence':payload,'top_k':k})
        results=[];seen=set()
        for row in response.get('results',[]):
            eid=row.get('expert_id')
            citations=list(dict.fromkeys(c for c in row.get('citations',[]) if isinstance(c,str) and c in visible and visible[c]['expert_id']==eid))
            if eid in experts and eid not in seen and citations:
                results.append({'expert':experts[eid],'reason':row.get('reason',''),'missing':row.get('missing',[]),
                                'evidence':[visible[c] for c in citations]});seen.add(eid)
            if len(results)==k:break
        return {'query':query,'results':results,'candidate_ids':[r['expert']['id'] for r in retrieved['results']],
                'trace':self.helper.trace,'usage':self.helper.usage,
                'timing':{'run_seconds':time.perf_counter()-start,'retrieval':retrieved['timing']},
                'scope':'One model call after candidate-LM retrieval and lexical evidence selection. Citation ownership is checked against supplied records; source identity, individual contribution, semantic support and expert ability are not independently verified.'}
