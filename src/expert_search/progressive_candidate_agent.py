"""Progressive candidate retrieval and separately validated model explanation."""
from copy import deepcopy
import time

from .candidate_evidence_pipeline import CandidateEvidencePipeline
from .candidate_reasoning_agent import CandidateReasoningAgent


class _FrozenRetrieval:
    def __init__(self, result): self.result = result
    def search_reranked(self, query, k, candidates, evidence_per_expert): return self.result


class ProgressiveCandidateAgent:
    """Yield preliminary owned evidence, then a separate explanation result.

    Initial results are lexical candidates, not semantic support or an expert
    capability judgment. Explanation errors do not silently certify them.
    """
    def __init__(self, pipeline=None, client=None):
        self.pipeline = pipeline if pipeline is not None else CandidateEvidencePipeline()
        self.client = client

    def events(self, query, k=5, candidates=20, rerank_pool=50, evidence_per_expert=1):
        if not isinstance(query,str) or not query.strip() or type(k)!=int or type(candidates)!=int or type(rerank_pool)!=int or not 1<=k<=candidates<=rerank_pool<=50:
            raise ValueError('Require a nonempty query and 1 <= k <= candidates <= rerank_pool <= 50')
        if type(evidence_per_expert)!=int or not 1<=evidence_per_expert<=3:
            raise ValueError('Require 1 to 3 evidence items per expert')
        start=time.perf_counter()
        retrieval=self.pipeline.search_reranked(query,k=candidates,candidates=rerank_pool,evidence_per_expert=evidence_per_expert)
        rows=retrieval['results'];seen_experts=set();seen_sources=set()
        for row in rows:
            expert=row['expert']['id']
            if expert in seen_experts:raise ValueError('Duplicate retrieved expert')
            seen_experts.add(expert)
            for evidence in row['evidence']:
                if evidence['expert_id']!=expert or evidence['id'] in seen_sources:raise ValueError('Invalid owned evidence')
                seen_sources.add(evidence['id'])
        yield {'stage':'retrieval', 'results':deepcopy(rows[:k]), 'seconds':time.perf_counter()-start,
               'source_snapshot_sha256':retrieval['source_snapshot_sha256'],
               'scope':'Preliminary lexical candidates with owned source records; not semantic support, individual contribution, or verified expert ability.'}
        if not seen_sources:
            yield {'stage':'explanation_unavailable','seconds':time.perf_counter()-start,'reason':'no_candidate_evidence',
                   'scope':'No model explanation was generated; missing evidence is not proof of inability.'}
            return
        try:
            agent=CandidateReasoningAgent(pipeline=_FrozenRetrieval(retrieval),client=self.client)
            result=agent.run(query,k=k,candidates=candidates,evidence_per_expert=evidence_per_expert,retrieval='adaptive_bm25',rerank_pool=rerank_pool)
        except Exception as error:
            yield {'stage':'explanation_failed','seconds':time.perf_counter()-start,'error_type':type(error).__name__,
                   'scope':'Preliminary retrieval remains unverified; no explanation or expert ability is inferred.'}
            return
        yield {'stage':'explanation','results':result['results'],'completion':result['completion'],'usage':result['usage'],
               'seconds':time.perf_counter()-start,
               'scope':'Owned exact quotes are structurally checked; explanation entailment and individual contribution remain unverified.'}
