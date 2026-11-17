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

    def run(self,query,k=5,candidates=20,evidence_per_expert=1,retrieval='lm',rerank_pool=50):
        if not isinstance(query,str) or not query.strip() or type(k)!=int or type(candidates)!=int or not 1<=k<=candidates<=50:
            raise ValueError('Require a nonempty query and 1 <= k <= candidates <= 50')
        if type(evidence_per_expert)!=int or not 1<=evidence_per_expert<=3:raise ValueError('Require 1 to 3 evidence items per expert')
        if retrieval not in ('lm','adaptive_bm25'):raise ValueError('Unknown retrieval strategy')
        if retrieval=='adaptive_bm25' and (type(rerank_pool)!=int or not candidates<=rerank_pool<=50):
            raise ValueError('Require candidates <= rerank_pool <= 50')
        start=time.perf_counter();self.helper.trace=[];self.helper.usage=dict.fromkeys(self.helper.usage,0)
        retrieved=(self.pipeline.search(query,candidates,evidence_per_expert) if retrieval=='lm' else
                   self.pipeline.search_reranked(query,k=candidates,candidates=rerank_pool,evidence_per_expert=evidence_per_expert))
        payload=[];visible={};experts={}
        for row in retrieved['results']:
            eid=row['expert']['id'];experts[eid]=row['expert'];items=[]
            for e in row['evidence']:
                if e['expert_id']!=eid or e['id'] in visible:raise ValueError('Invalid evidence ownership or duplicate evidence ID')
                item={'id':e['id'],'expert_id':eid,'text':e['text'][:6000],
                      'kind':e.get('kind','unknown'), 'publication':e.get('publication'),
                      'contribution_status':'unverified',
                      'authorship_association':'source_association_not_independently_verified'}
                if 'citation_context' in e:
                    context=e['citation_context']
                    required={'observed_total','observation_date','total_history_status','field_normalized_impact','personal_credit'}
                    if (e.get('kind')!='paper' or not isinstance(context,dict) or set(context)!=required or
                        (context['observed_total'] is not None and (type(context['observed_total']) is not int or context['observed_total']<0)) or
                        context['observation_date'] is not None or context['field_normalized_impact'] is not None or context['personal_credit'] is not None or
                        context['total_history_status'] not in ('unknown_total','yearly_sum_exceeds_total','yearly_sum_matches_total','partial_or_unresolved_history')):
                        raise ValueError('Invalid paper citation context')
                    item['citation_context']=dict(context)
                visible[e['id']]=item;items.append(item)
            if items:payload.extend(items)
        if not payload:raise ValueError('No candidate evidence')
        citation_instruction=' Citation totals, when supplied, are unnormalized paper-level observations of unknown date; never infer individual credit from them.' if any('citation_context' in e for e in payload) else ''
        response=self.helper.ask('verify','Rank experts using only supplied evidence. For each claim cite an owned evidence ID and an exact nonempty quote copied from its visible text; state missing requirements. Paper ownership is an unverified association, not proof of individual contribution or method credit. Respect publication dates when present and state when date or contribution is unknown. Do not infer capability from names or institutions. Return {"results":[{"expert_id":str,"reason":str,"missing":[str],"citations":[{"evidence_id":str,"quote":str}]}]}.'+citation_instruction,{'demand':query,'evidence':payload,'top_k':k})
        results=[];seen=set();rejected=0
        rows=response.get('results',[])
        if not isinstance(rows,list):raise ValueError('Invalid result list')
        for row in rows:
            if not isinstance(row,dict):
                rejected+=1;continue
            eid=row.get('expert_id');reason=row.get('reason');missing=row.get('missing');citations=row.get('citations')
            if (not isinstance(eid,str) or eid not in experts or eid in seen or not isinstance(reason,str) or not reason.strip() or
                    not isinstance(missing,list) or any(not isinstance(x,str) for x in missing) or not isinstance(citations,list)):
                rejected+=1;continue
            anchored=[];used=set()
            for citation in citations:
                if not isinstance(citation,dict):continue
                cid=citation.get('evidence_id');quote=citation.get('quote')
                if (not isinstance(cid,str) or cid in used or cid not in visible or visible[cid]['expert_id']!=eid or
                        not isinstance(quote,str) or not quote.strip()):continue
                source=visible[cid];start_offset=source['text'].find(quote)
                if start_offset<0:continue
                anchored.append(dict(source,quote=quote,start=start_offset,end=start_offset+len(quote)))
                used.add(cid)
            if not anchored:
                rejected+=1;continue
            results.append({'expert':experts[eid],'reason':reason,'missing':missing,'evidence':anchored});seen.add(eid)
            if len(results)==k:break
        completion = {'requested':k, 'returned':len(results), 'rejected':rejected,
                      'status':'empty' if not results else 'complete' if len(results)==k else 'partial',
                      'scope':'Structural count only; empty may be an abstention and complete does not establish relevance or evidence entailment.'}
        return {'query':query,'retrieval_strategy':retrieval,'results':results,'candidate_ids':[r['expert']['id'] for r in retrieved['results']],
                'completion':completion,
                'trace':self.helper.trace,'usage':self.helper.usage,'rejected_results':rejected,
                'timing':{'run_seconds':time.perf_counter()-start,'retrieval':retrieved['timing']},
                'scope':'One model call after candidate-LM retrieval and lexical evidence selection. Citation ownership and exact quote offsets are checked against visible supplied records; source identity, individual contribution, semantic support and expert ability are not independently verified.'}
