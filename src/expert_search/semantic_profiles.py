"""Frozen semantic transport of profile and background term distributions.

Translation language models are prior work. This prototype tests compiling a
fixed sparse semantic kernel into expert-profile likelihood contributions.
"""
import math
from collections import defaultdict
import numpy as np
from .expert_language_models import ExpertLanguageModels


def mutual_kernel(words,vectors,k=8,self_mass=.5):
    if len(set(words))!=len(words) or len(vectors)!=len(words) or type(k)!=int or k<1 or not 0<self_mass<=1:
        raise ValueError('Invalid semantic kernel inputs')
    vectors=np.asarray(vectors)
    if vectors.ndim!=2 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Require finite unit embeddings')
    if len(words)<2:return {word:{word:1.} for word in words}
    degree=min(k,len(words)-1)
    nearest=[]
    for start in range(0,len(words),256):
        scores=vectors[start:start+256]@vectors.T
        for offset,row in enumerate(scores):
            index=start+offset
            row[index]=-np.inf
            threshold=np.partition(row,len(words)-degree)[len(words)-degree]
            possible=np.flatnonzero(row>=threshold)
            ordered=sorted(possible,key=lambda j:(-float(row[j]),words[j]))[:degree]
            nearest.append(set(map(int,ordered)))
    kernel={}
    for index,word in enumerate(words):
        neighbors=sorted(j for j in nearest[index] if index in nearest[j])
        kernel[word]=({word:1.} if not neighbors else
                      {word:self_mass,**{words[j]:(1-self_mass)/len(neighbors) for j in neighbors}})
    return kernel


class TransportedProfiles:
    MODES=('model1-document',)

    def __init__(self,reference,kernel):
        self.experts=reference.experts
        self.members=reference.members
        self.smoothing=reference.smoothing
        vocabulary=set(reference.background)
        if set(kernel)-vocabulary:raise ValueError('Kernel contains unknown source terms')
        for word,row in kernel.items():
            if (not row or set(row)-vocabulary or word not in row or row[word]<=0 or
                    any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in row.values()) or
                    not math.isclose(math.fsum(row.values()),1,abs_tol=1e-10)):
                raise ValueError('Kernel rows must preserve probability mass and self support')
        def transport(distribution):
            output=defaultdict(float)
            for word in sorted(distribution):
                for target,weight in kernel.get(word,{word:1.}).items():
                    if weight:output[target]+=distribution[word]*weight
            if not math.isclose(math.fsum(output.values()),math.fsum(distribution.values()),abs_tol=1e-8):
                raise ValueError('Transport lost probability mass')
            return dict(output)
        self.background=transport(reference.background)
        self.profile_terms={eid:transport(terms) for eid,terms in reference.profile_terms.items()}

    def score(self,query,eligible=None):
        return ExpertLanguageModels.score(self,query,mode='model1-document',eligible=eligible)
