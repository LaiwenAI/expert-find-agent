"""Attach verified-snapshot publication metadata without modifying retrieval."""
from copy import deepcopy
import hashlib
from itertools import zip_longest
from pathlib import Path

from .compiled_expert_search import corpus_snapshot
from .corpus import records
from .metadata.papers import publication_interval


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MetadataOverlayPipeline:
    """Add supplied date metadata to an unchanged retrieval pipeline.

    Every original field and record order must match the indexed corpus. The
    overlay stores source hashes and cannot establish date truth, attribution,
    historical availability, or semantic relevance; it does not filter by time.
    """
    def __init__(self,pipeline,base_corpus,enriched_corpus):
        base=Path(base_corpus);enriched=Path(enriched_corpus)
        if corpus_snapshot(base)!=pipeline.ranker.snapshot:raise ValueError('Pipeline and base corpus differ')
        if file_sha(base/'experts.jsonl')!=file_sha(enriched/'experts.jsonl'):raise ValueError('Expert records differ')
        mapping={};allowed={'publication','contribution_status','authorship_association'}
        for left,right in zip_longest(records(base/'evidence.jsonl'),records(enriched/'evidence.jsonl')):
            if left is None or right is None:raise ValueError('Evidence count mismatch')
            _,original=left;_,expanded=right;eid=original['id']
            if eid in mapping or any(k not in expanded or expanded[k]!=v for k,v in original.items()):raise ValueError('Changed or duplicated evidence')
            if set(expanded)-set(original)-allowed:raise ValueError('Unexpected metadata fields')
            if expanded.get('contribution_status') not in (None,'unverified'):raise ValueError('Unverified role required')
            if expanded.get('authorship_association') not in (None,'provided_teacher_id_not_independently_verified','source_association_not_independently_verified'):
                raise ValueError('Unverified source association required')
            publication=expanded.get('publication')
            if publication is not None and (not isinstance(publication,dict) or publication!=publication_interval(publication.get('raw'))):
                raise ValueError('Invalid publication interval')
            extras={k:deepcopy(expanded[k]) for k in allowed if k in expanded}
            mapping[eid]={'owner':original['expert_id'],'text_sha256':hashlib.sha256(original['text'].encode()).hexdigest(),'extras':extras}
        self.pipeline=pipeline;self.mapping=mapping;self.base_snapshot_sha256=pipeline.ranker.snapshot
        self.enriched_evidence_sha256=file_sha(enriched/'evidence.jsonl')

    def _attach(self,result):
        result=deepcopy(result)
        if result.get('source_snapshot_sha256')!=self.base_snapshot_sha256:raise ValueError('Changed retrieval snapshot')
        for row in result['results']:
            owner=row['expert']['id']
            for evidence in row['evidence']:
                metadata=self.mapping.get(evidence['id'])
                if (metadata is None or metadata['owner']!=owner or evidence.get('expert_id')!=owner or
                    hashlib.sha256(evidence['text'].encode()).hexdigest()!=metadata['text_sha256']):raise ValueError('Changed source evidence')
                for key,value in metadata['extras'].items():
                    if key in evidence and evidence[key]!=value:raise ValueError('Conflicting retrieved metadata')
                    evidence[key]=deepcopy(value)
        result['metadata_overlay']={'base_snapshot_sha256':self.base_snapshot_sha256,'enriched_evidence_sha256':self.enriched_evidence_sha256,
            'scope':'Supplied publication metadata and unverified source association; no date truth, personal contribution or historical availability verification. No temporal filtering.'}
        return result

    def search(self,*args,**kwargs):return self._attach(self.pipeline.search(*args,**kwargs))
    def search_reranked(self,*args,**kwargs):return self._attach(self.pipeline.search_reranked(*args,**kwargs))
