"""Exact inverted compilation of a classical fixed-smoothing candidate LM.

Engineering optimization of an existing additive model, not a novel ranker.
Precompute log contributions, visit only nonzero postings, retain math.fsum so
the reference scorer's floating-point outputs and ties are preserved.
"""
import math
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .retrieval import tokens


class CompiledCandidateLM:
    def __init__(self, reference):
        self.experts = tuple(reference.experts)
        self.available = frozenset(e for e in self.experts if reference.members[e])
        self.positions = {eid:i for i,eid in enumerate(self.experts)}
        self.background = frozenset(reference.background)
        ratio = (1-reference.smoothing)/reference.smoothing
        postings = defaultdict(list)
        for eid in self.experts:
            for term,probability in reference.profile_terms[eid].items():
                value = math.log1p(ratio*probability/reference.background[term])
                if value:
                    postings[term].append((self.positions[eid], value))
        self.postings = {term:(np.asarray([i for i,_ in pairs],dtype=np.int32),
                              np.asarray([v for _,v in pairs],dtype=np.float64)) for term,pairs in postings.items()}
        self.array_bytes = sum(indices.nbytes+values.nbytes for indices,values in self.postings.values())
        self.nonzero_postings = sum(len(indices) for indices,_ in self.postings.values())

    def score(self, query, eligible=None):
        candidates = sorted(self.available) if eligible is None else sorted(eligible)
        if set(candidates)-self.available:
            raise ValueError('Candidate missing or has no nonempty source')
        contributions = [[] for _ in self.experts]
        for term,count in Counter(tokens(query)).items():
            posting = self.postings.get(term)
            if posting is not None:
                indices,values = posting
                for index,value in zip(indices,values):
                    contributions[index].append(count*float(value))
        return {eid:math.fsum(contributions[self.positions[eid]]) for eid in candidates}

    def save(self, directory, source_sha256):
        """Persist numeric arrays without pickle; bind them to a corpus snapshot."""
        out = Path(directory)
        out.mkdir(parents=True,exist_ok=True)
        terms = sorted(self.postings)
        offsets = np.zeros(len(terms)+1,dtype=np.int64)
        for i,term in enumerate(terms):
            offsets[i+1] = offsets[i]+len(self.postings[term][0])
        indices = np.concatenate([self.postings[t][0] for t in terms]) if terms else np.array([],dtype=np.int32)
        values = np.concatenate([self.postings[t][1] for t in terms]) if terms else np.array([],dtype=np.float64)
        partial = out/'arrays.partial.npz'
        np.savez_compressed(partial,offsets=offsets,indices=indices,values=values)
        partial.replace(out/'arrays.npz')
        metadata = dict(version=1,experts=list(self.experts),available=sorted(self.available),terms=terms,
                        source_sha256=source_sha256,arrays_sha256=hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest())
        partial = out/'metadata.partial.json'
        partial.write_text(json.dumps(metadata,ensure_ascii=False))
        partial.replace(out/'metadata.json')

    @classmethod
    def load(cls, directory, expected_source_sha256):
        out = Path(directory)
        metadata = json.loads((out/'metadata.json').read_text())
        if metadata.get('version') != 1 or metadata.get('source_sha256') != expected_source_sha256:
            raise ValueError('Changed compiled index source or format')
        if hashlib.sha256((out/'arrays.npz').read_bytes()).hexdigest() != metadata['arrays_sha256']:
            raise ValueError('Corrupt compiled index arrays')
        terms,experts,available = metadata['terms'],metadata['experts'],metadata['available']
        if (not all(isinstance(s,str) and s for s in terms+experts+available) or
                terms != sorted(set(terms)) or experts != sorted(set(experts)) or
                len(available) != len(set(available)) or set(available)-set(experts)):
            raise ValueError('Invalid compiled index identifiers')
        with np.load(out/'arrays.npz',allow_pickle=False) as arrays:
            offsets,indices,values = (arrays[k] for k in ('offsets','indices','values'))
        if (offsets.dtype != np.int64 or indices.dtype != np.int32 or values.dtype != np.float64 or
                offsets.ndim != 1 or indices.ndim != 1 or values.ndim != 1 or
                len(offsets) != len(terms)+1 or offsets[0] != 0 or offsets[-1] != len(indices) or
                len(indices) != len(values) or (np.diff(offsets)<0).any() or
                (indices<0).any() or (indices>=len(experts)).any() or
                not np.isfinite(values).all() or (values<0).any()):
            raise ValueError('Invalid compiled index numeric arrays')
        result = cls.__new__(cls)
        result.experts,result.available,result.background = tuple(experts),frozenset(available),frozenset(terms)
        result.positions = {eid:i for i,eid in enumerate(experts)}
        indices.flags.writeable = False
        values.flags.writeable = False
        result.postings = {}
        for i,term in enumerate(terms):
            start,end = offsets[i:i+2]
            if (np.diff(indices[start:end])<=0).any():
                raise ValueError('Duplicate or unordered expert postings')
            result.postings[term] = (indices[start:end],values[start:end])
        result.array_bytes = indices.nbytes+values.nbytes
        result.nonzero_postings = len(indices)
        return result
