"""Attach traceable, unnormalized paper citation observations as context."""
from copy import deepcopy
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from .compiled_expert_search import corpus_snapshot
from .corpus import records
from .metadata.papers import citation_metadata
from .metadata_overlay import file_sha


class CitationOverlayPipeline:
    """Bind the original paper metadata sidecar to the indexed source snapshot.

    This provides display context only. Observation dates, self-citations, field
    normalization and personal credit remain unknown; no time filter or score
    modification is performed.
    """
    def __init__(self,pipeline,base_corpus,sidecar,raw_papers):
        base=Path(base_corpus);sidecar=Path(sidecar);raw_papers=Path(raw_papers)
        snapshot=getattr(pipeline,'base_snapshot_sha256',None) or pipeline.ranker.snapshot
        if corpus_snapshot(base)!=snapshot:raise ValueError('Pipeline and base corpus differ')
        manifest=json.loads((sidecar/'manifest.json').read_text());metadata=sidecar/'papers.jsonl'
        if file_sha(raw_papers)!=manifest['source_sha256'] or file_sha(metadata)!=manifest['metadata_sha256']:raise ValueError('Changed citation source or sidecar')
        papers={r['id']:r for _,r in records(base/'evidence.jsonl') if r.get('kind')=='paper'}
        if len(papers)!=sum(1 for _,r in records(base/'evidence.jsonl') if r.get('kind')=='paper'):raise ValueError('Duplicate paper IDs')
        contexts={};count=0
        for source,meta_row in zip_longest(records(raw_papers),records(metadata)):
            if source is None or meta_row is None:raise ValueError('Citation record count mismatch')
            line,raw=source;_,meta=meta_row;count+=1;source_id='paper:'+str(line)
            raw_hash=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if (meta.get('source_id')!=source_id or meta.get('raw_line')!=line or meta.get('expert_id')!=raw.get('teacher_id') or
                meta.get('raw_record_sha256')!=raw_hash or meta.get('citations')!=citation_metadata(raw)):raise ValueError('Citation sidecar record mismatch')
            if source_id not in papers:continue
            paper=papers[source_id];text='\n'.join(str(raw.get(k) or '') for k in ('title','gs_abstract'))
            if paper['expert_id']!=raw['teacher_id'] or paper['text']!=text or paper.get('title')!=raw.get('title'):
                raise ValueError('Indexed paper and citation source differ')
            c=meta['citations'];contexts[source_id]={'owner':paper['expert_id'],'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'context':{'observed_total':c['observed_total'],'observation_date':None,'total_history_status':c['total_history_status'],
                           'field_normalized_impact':None,'personal_credit':None}}
        if count!=manifest['rows'] or set(contexts)!=set(papers):raise ValueError('Incomplete citation coverage')
        self.pipeline=pipeline;self.base_snapshot_sha256=snapshot;self.contexts=contexts
        self.sidecar_sha256=file_sha(metadata);self.raw_papers_sha256=file_sha(raw_papers)

    def _attach(self,result):
        result=deepcopy(result)
        if result.get('source_snapshot_sha256')!=self.base_snapshot_sha256:raise ValueError('Changed retrieval snapshot')
        for row in result['results']:
            for evidence in row['evidence']:
                if evidence.get('kind')!='paper':continue
                context=self.contexts.get(evidence['id'])
                if (context is None or context['owner']!=row['expert']['id'] or evidence.get('expert_id')!=context['owner'] or
                    hashlib.sha256(evidence['text'].encode()).hexdigest()!=context['text_sha256'] or 'citation_context' in evidence):raise ValueError('Changed or ambiguous paper evidence')
                evidence['citation_context']=deepcopy(context['context'])
        result['citation_overlay']={'sidecar_sha256':self.sidecar_sha256,'raw_papers_sha256':self.raw_papers_sha256,
            'scope':'Unnormalized paper-level citation observations with unknown observation date and personal credit; not a relevance or retrospective impact score.'}
        return result

    def search(self,*args,**kwargs):return self._attach(self.pipeline.search(*args,**kwargs))
    def search_reranked(self,*args,**kwargs):return self._attach(self.pipeline.search_reranked(*args,**kwargs))
