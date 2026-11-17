"""Exact candidate-LM shortlist followed by candidate-first lexical evidence lookup."""
import itertools
import json
import time
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
                 evidence_database='artifacts/fast-search-selective-v1.sqlite'):
        start=time.perf_counter()
        self.ranker=CompiledExpertSearch(corpus,language_index)
        self.evidence=FastSearch(corpus,evidence_database)
        verify_evidence_documents(corpus,self.evidence)
        for table,name in [('docs','docs_expert_idx'),('scores','scores_doc_term_idx')]:
            if name not in {row[1] for row in self.evidence.db.execute(f'PRAGMA index_list({table})')}:
                raise ValueError('Missing candidate-first index: '+name)
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
