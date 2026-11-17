"""Exact candidate-LM shortlist followed by candidate-first lexical evidence lookup."""
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

from .compiled_expert_search import CompiledExpertSearch
from .corpus import records
from .fast_retrieval import FastSearch


def verify_evidence_documents(corpus,searcher):
    """Match compiled document IDs, bodies and owners to the supplied corpus."""
    corpus_rows=enumerate(records(Path(corpus)/'evidence.jsonl'))
    database_rows=searcher.db.execute('SELECT id,body,evidence_id,expert_id FROM docs ORDER BY id')
    for source,stored in itertools.zip_longest(corpus_rows,database_rows):
        if source is None or stored is None:raise ValueError('Evidence document count mismatch')
        index,(_,row)=source;doc,body,evidence_id,expert_id=stored
        if doc!=index or json.loads(body)!=row or evidence_id!=row['id'] or expert_id!=row['expert_id']:
            raise ValueError('Evidence corpus and compiled documents differ')


class CandidateEvidencePipeline:
    def __init__(self,corpus='artifacts/corpus',language_index='artifacts/teacher-compiled-lm',
                 evidence_database='artifacts/fast-search-selective-v1.sqlite',scoring='sqlite'):
        if scoring not in ('sqlite','resident'):
            raise ValueError('Require sqlite or resident scoring')
        start=time.perf_counter()
        self.ranker=CompiledExpertSearch(corpus,language_index)
        self.evidence=FastSearch(corpus,evidence_database)
        verify_evidence_documents(corpus,self.evidence)
        for table,name in [('docs','docs_expert_idx'),('scores','scores_doc_term_idx')]:
            if name not in {row[1] for row in self.evidence.db.execute(f'PRAGMA index_list({table})')}:
                raise ValueError('Missing candidate-first index: '+name)
        self.evidence_by_expert=defaultdict(list)
        for _,(evidence_id,expert_id) in self.evidence.metadata.items():
            self.evidence_by_expert[expert_id].append(evidence_id)
        self.resident = None
        if scoring == 'resident':
            from .resident_candidate_bm25 import ResidentCandidateBM25
            self.resident = ResidentCandidateBM25(evidence_database)
            expected = tuple(self.evidence.metadata[doc][0] for doc in sorted(self.evidence.metadata))
            if self.resident.ids != expected:
                raise ValueError('Resident evidence snapshot mismatch')
        self.initialization_seconds=time.perf_counter()-start

    def search(self,query,k=5,evidence_per_expert=3):
        start=time.perf_counter();ranked=self.ranker.search(query,k)
        ids=[row['expert']['id'] for row in ranked['results']]
        evidence_start=time.perf_counter()
        evidence=self.evidence.evidence_for_selective(query,ids,evidence_per_expert)
        evidence_seconds=time.perf_counter()-evidence_start
        results=[dict(row,evidence=evidence[row['expert']['id']]) for row in ranked['results']]
        return {'results':results,'timing':{'ranking_seconds':ranked['seconds'],'evidence_seconds':evidence_seconds,
                'total_seconds':time.perf_counter()-start,'initialization_seconds':self.initialization_seconds},
                'source_snapshot_sha256':ranked['source_snapshot_sha256'],
                'scope':'Candidate language-model ranking followed by lexical evidence selection; no model reasoning or relevance validation. Missing evidence does not establish inability.'}

    def search_reranked(self,query,k=10,candidates=50,evidence_per_expert=1,aggregation='max'):
        """Candidate-LM shortlist, adaptive BM25 aggregation, then owned evidence.

        Vote sums the five highest source scores and omits zero-score experts;
        it may use more scoring sources than the requested displayed evidence.
        Max retains zero-score fallback results. Scores and evidence reflect the supplied frozen corpus only. A zero
        score can still yield fallback evidence; neither rank nor association
        proves semantic support, identity, personal role, or expert ability.
        """
        if aggregation not in ('max','vote'):
            raise ValueError('Require max or vote aggregation')
        if type(k)!=int or type(candidates)!=int or not 1<=k<=candidates<=50:
            raise ValueError('Require 1 <= k <= candidates <= 50')
        if type(evidence_per_expert)!=int or not 1<=evidence_per_expert<=10:
            raise ValueError('Require 1 to 10 evidence items per expert')
        start=time.perf_counter()
        candidates_result=self.ranker.search(query,candidates)
        pool=[row['expert']['id'] for row in candidates_result['results']]
        expert_rows={row['expert']['id']:row['expert'] for row in candidates_result['results']}
        source_ids=[eid for expert in pool for eid in self.evidence_by_expert[expert]]
        scoring_start=time.perf_counter()
        routed=(self.evidence.adaptive_document_scores(query,source_ids) if self.resident is None else
                dict(scores=self.resident.selected_scores(query,source_ids),route='resident',
                     estimated_global_postings=None,estimated_selective_probes=None))
        scores={}
        for expert in pool:
            values=[routed['scores'][eid] for eid in self.evidence_by_expert[expert]]
            scores[expert]=max(values,default=0.) if aggregation=='max' else sum(sorted(values,reverse=True)[:5])
        # Vote follows the lexical baseline: candidates with no matching source
        # are not returned; max retains its existing zero-score fallback.
        eligible=pool if aggregation=='max' else [expert for expert in pool if scores[expert]>0]
        ranking=sorted(eligible,key=lambda expert:(-scores[expert],expert))[:k]
        scoring_seconds=time.perf_counter()-scoring_start
        evidence_start=time.perf_counter()
        evidence=self.evidence.evidence_from_scores(ranking,self.evidence_by_expert,routed['scores'],evidence_per_expert)
        evidence_seconds=time.perf_counter()-evidence_start
        result=[dict(expert=expert_rows[expert],score=scores[expert],evidence=evidence[expert]) for expert in ranking]
        return dict(results=result,source_snapshot_sha256=candidates_result['source_snapshot_sha256'],
                    aggregation=aggregation,route=routed['route'],estimated_global_postings=routed['estimated_global_postings'],
                    estimated_selective_probes=routed['estimated_selective_probes'],
                    timing=dict(ranking_seconds=candidates_result['seconds'],scoring_seconds=scoring_seconds,
                                evidence_seconds=evidence_seconds,total_seconds=time.perf_counter()-start,
                                initialization_seconds=self.initialization_seconds),
                    scope='Candidate-LM shortlist and adaptive BM25 aggregation followed by owned evidence. Max may return zero-score fallback evidence; vote omits zero-score candidates. Association, source semantics, individual role and ability remain unverified.')
