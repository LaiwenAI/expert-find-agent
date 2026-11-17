"""Candidate-first expert retrieval with structured, source-bound method hypotheses."""
import hashlib
import time

from .candidate_evidence_pipeline import CandidateEvidencePipeline
from .partial_method_match import PartialMethodMatcher


class StructuredCandidateAgent:
    """Run the existing partial-method matcher on bounded retrieved source views.

    Output spans, labels and mechanisms are model proposals. Structural checks
    do not establish entailment, expert identity, individual credit or success.
    No missing evidence is interpreted as expert inability.
    """
    def __init__(self,pipeline=None,client=None,matcher=None):
        self.pipeline=pipeline if pipeline is not None else CandidateEvidencePipeline()
        if matcher is None:
            if client is None:
                from laiwenai import LaiwenAI
                client=LaiwenAI(model='gpt-6-luna')
            matcher=PartialMethodMatcher(client)
        self.matcher=matcher

    def run(self,query,k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200):
        if not isinstance(query,str) or not query.strip() or any(type(v)!=int for v in (k,candidates,rerank_pool,evidence_per_expert,source_chars)):
            raise ValueError('Invalid query or budgets')
        if not 1<=k<=candidates<=8 or not candidates<=rerank_pool<=50 or not 1<=evidence_per_expert<=3 or not 1<=source_chars<=6000:
            raise ValueError('Invalid query or budgets')
        start=time.perf_counter();retrieved=self.pipeline.search_reranked(query,k=candidates,candidates=rerank_pool,evidence_per_expert=evidence_per_expert)
        source_views={};expert_records={};seen=set()
        for row in retrieved['results']:
            expert=row['expert'];eid=expert['id']
            if eid in expert_records:raise ValueError('Duplicate expert')
            expert_records[eid]=expert;source_views[eid]={}
            for e in row['evidence']:
                if e['expert_id']!=eid or e['id'] in seen or not isinstance(e.get('text'),str):raise ValueError('Invalid source ownership')
                seen.add(e['id']);text=e['text'][:source_chars]
                source_views[eid][e['id']]={'id':e['id'],'expert_id':eid,'kind':e.get('kind','unknown'),'text':text}
        if not expert_records or not any(s['text'].strip() for views in source_views.values() for s in views.values()):
            return {'status':'unavailable','reason':'no_candidate_evidence','results':[],'source_snapshot_sha256':retrieved['source_snapshot_sha256'],
                    'seconds':time.perf_counter()-start,'scope':'No model judgment; missing evidence is not expert inability.'}
        result=self.matcher.match(query,source_views,{},'raw')
        if result['status']!='structurally_valid':
            return {'status':result['status'],'results':[],'matcher':result,'source_snapshot_sha256':retrieved['source_snapshot_sha256'],
                    'seconds':time.perf_counter()-start,'scope':'No completed ranking; failure or missing source evidence does not imply inability.'}
        ranking=result['result']['ranking'];ids=[r['expert_id'] for r in ranking]
        if len(ids)!=len(expert_records) or len(set(ids))!=len(ids) or set(ids)!=set(expert_records):raise ValueError('Invalid structured candidate inventory')
        for row in ranking:
            eid=row['expert_id']
            for c in row['demand_citations']:
                if type(c.get('start'))!=int or type(c.get('end'))!=int or not 0<=c['start']<c['end']<=len(query) or query[c['start']:c['end']]!=c['quote']:
                    raise ValueError('Invalid demand anchor')
            for ref in row['evidence']:
                if ref['expert_id']!=eid:raise ValueError('Foreign source reference')
                for c in ref['citations']:
                    source=source_views[eid].get(c['evidence_id']);a=c.get('start');b=c.get('end')
                    if source is None or type(a)!=int or type(b)!=int or not 0<=a<b<=len(source['text']) or source['text'][a:b]!=c['quote']:
                        raise ValueError('Invalid source span')
        selected=[dict(row,expert=expert_records[row['expert_id']]) for row in ranking[:k]]
        hashes={sid:hashlib.sha256(source['text'].encode()).hexdigest() for sources in source_views.values() for sid,source in sources.items()}
        return {'status':'structurally_valid','results':selected,'candidate_ids':ids,'source_view_sha256':hashes,
                'completion':{'requested':k,'returned':len(selected),'status':'complete' if len(selected)==k else 'partial'},
                'matcher':result,'source_snapshot_sha256':retrieved['source_snapshot_sha256'],'retrieval_timing':retrieved['timing'],
                'seconds':time.perf_counter()-start,'scope':'Structured model hypotheses with checked demand and owned source spans; method transfer, personal role, time and semantic support remain unverified.'}
