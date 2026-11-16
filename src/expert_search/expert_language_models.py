"""Classical Balog et al. (SIGIR 2006) expert language-model baselines.

Binary source associations, uniform candidate prior and fixed JM smoothing.
Model 1 averages normalized document term distributions (not concatenated TF).
Model 2 marginalizes document query likelihoods, computed in log space.
"""
import math
from collections import Counter, defaultdict
import numpy as np
from .retrieval import tokens


class ExpertLanguageModels:
    MODES = ('model1-document', 'model2-document', 'model2-candidate')

    def __init__(self, documents, profiles, smoothing=.5):
        if type(smoothing) not in (int, float) or not 0 < smoothing <= 1:
            raise ValueError('Require background interpolation in (0,1]')
        self.smoothing = smoothing
        counters = {pid: Counter(tokens(text)) for pid,text in documents.items()}
        self.empty_sources = sorted(pid for pid,c in counters.items() if not c)
        counters = {pid:c for pid,c in counters.items() if c}
        if not counters:
            raise ValueError('Empty token collection')
        self.docids = sorted(counters)
        position = {pid:i for i,pid in enumerate(self.docids)}
        if any(set(ids)-set(documents) for ids in profiles.values()):
            raise ValueError('Unknown source ID')
        self.members = {eid: sorted(position[pid] for pid in set(ids) if pid in position) for eid,ids in profiles.items()}
        self.experts = sorted(profiles)
        background = Counter()
        postings = defaultdict(list)
        for pid in self.docids:
            counts = counters[pid]
            background.update(counts)
            size = sum(counts.values())
            for term,count in counts.items():
                postings[term].append((position[pid],count/size))
        total = sum(background.values())
        self.background = {term:count/total for term,count in background.items()}
        self.postings = {term:(np.asarray([p[0] for p in pairs]),np.asarray([p[1] for p in pairs])) for term,pairs in postings.items()}
        self.profile_terms = {}
        for eid,indices in self.members.items():
            distribution = defaultdict(float)
            for i in indices:
                counts = counters[self.docids[i]]
                size = sum(counts.values())
                for term,count in counts.items():
                    distribution[term] += count/(size*len(indices))
            self.profile_terms[eid] = distribution
        degrees = Counter(i for indices in self.members.values() for i in indices)
        self.log_degree = np.asarray([math.log(degrees[i]) if degrees[i] else 0. for i in range(len(self.docids))])

    def score(self, query, mode='model1-document', eligible=None):
        if mode not in self.MODES:
            raise ValueError('Unknown language-model mode')
        candidates = self.experts if eligible is None else sorted(eligible)
        if any(eid not in self.members or not self.members[eid] for eid in candidates):
            raise ValueError('Candidate missing or has no nonempty source')
        query_counts = {term:count for term,count in Counter(tokens(query)).items() if term in self.background}
        ratio = (1-self.smoothing)/self.smoothing
        # Query-only log P(q|lambda*background) is common to all candidates.
        if mode == 'model1-document':
            return {eid:math.fsum(count*math.log1p(ratio*self.profile_terms[eid].get(term,0.)/self.background[term])
                                 for term,count in query_counts.items()) for eid in candidates}
        document_scores = np.zeros(len(self.docids), dtype=float)
        for term,count in query_counts.items():
            indices,probabilities = self.postings[term]
            document_scores[indices] += count*np.log1p(ratio*probabilities/self.background[term])
        if mode == 'model2-candidate':
            document_scores -= self.log_degree
        scores = {}
        for eid in candidates:
            values = document_scores[self.members[eid]]
            maximum = float(values.max())
            value = maximum+math.log(float(np.exp(values-maximum).sum()))
            if mode == 'model2-document':
                value -= math.log(len(values))
            scores[eid] = value
        return scores
