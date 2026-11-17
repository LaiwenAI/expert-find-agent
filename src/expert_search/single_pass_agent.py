"""Single-call evidence ranking control using a frozen lexical candidate pool."""
import time
from .agent import ExpertAgent


class SinglePassAgent(ExpertAgent):
    def run(self, query, k=5, candidates=20, exclude=()):
        if not query.strip() or not 1 <= k <= candidates <= 50:
            raise ValueError('Require a nonempty query and 1 <= k <= candidates <= 50')
        start=time.perf_counter()
        self.trace=[];self.usage=dict.fromkeys(self.usage,0)
        retrieval_start=time.perf_counter()
        hits=self.searcher.search(query,candidates,exclude=exclude)
        retrieval_seconds=time.perf_counter()-retrieval_start
        evidence={e['id']:e for hit in hits for e in hit['evidence']}
        payload=[{'expert_id':hit['id'],'evidence':[{'id':e['id'],'text':e['text'][:6000]} for e in hit['evidence']]} for hit in hits]
        response=self.ask('verify','Rank experts using only supplied evidence. Cite evidence IDs and state missing requirements. Do not infer capabilities from names or institutions. Schema: {"results":[{"expert_id":str,"reason":str,"missing":[str],"citations":[str]}]}. Omit experts without supporting evidence.',{'demand':query,'candidates':payload,'top_k':k})
        allowed={hit['id'] for hit in hits};results=[];seen=set()
        for row in response.get('results',[]):
            eid=row.get('expert_id')
            cites=[cid for cid in row.get('citations',[]) if isinstance(cid,str) and cid in evidence and evidence[cid]['expert_id']==eid]
            if eid in allowed and eid not in seen and cites:
                seen.add(eid);results.append(dict(expert=self.searcher.experts[eid],reason=row.get('reason',''),missing=row.get('missing',[]),evidence=[evidence[cid] for cid in dict.fromkeys(cites)]))
            if len(results)==k:break
        return dict(query=query,results=results,trace=self.trace,usage=self.usage,candidate_ids=[x['id'] for x in hits],
                    timing={'run_seconds':time.perf_counter()-start,'retrieval_seconds':retrieval_seconds,'scope':'Run call only; excludes initialization and index construction.'},
                    method='single-pass-evidence-control',notice='Citation ownership is validated; relevance is model-generated.')
