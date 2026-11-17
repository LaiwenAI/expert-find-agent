"""Persistent classical candidate-LM retrieval over the local teacher corpus."""
import hashlib
import json
import time
from pathlib import Path
from .compiled_language_model import CompiledCandidateLM
from .certified_topk import certified_topk
from .corpus import records


def corpus_snapshot(corpus):
    root = Path(corpus)
    digests = [hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('evidence.jsonl','experts.jsonl')]
    return hashlib.sha256(''.join(digests).encode()).hexdigest()


class CompiledExpertSearch:
    def __init__(self, corpus='artifacts/corpus', index='artifacts/teacher-compiled-lm'):
        start = time.perf_counter()
        self.snapshot = corpus_snapshot(corpus)
        self.model = CompiledCandidateLM.load(index,self.snapshot)
        self.experts = {r['id']:r for _,r in records(Path(corpus)/'experts.jsonl')}
        if set(self.model.experts)-set(self.experts):
            raise ValueError('Compiled index has unknown expert IDs')
        self.initialization_seconds = time.perf_counter()-start

    def search(self, query, k=5, *, _tokens=None):
        if not isinstance(query,str) or not query.strip() or type(k)!=int or not 1<=k<=50:
            raise ValueError('Require nonempty query and integer k in [1,50]')
        start = time.perf_counter()
        selected = certified_topk(self.model,query,k,_tokens=_tokens)
        results = [dict(expert=self.experts[row['id']],score=row['score']) for row in selected['results']]
        elapsed = time.perf_counter()-start
        return dict(results=results,seconds=elapsed,initialization_seconds=self.initialization_seconds,
                    source_snapshot_sha256=self.snapshot,total_experts=len(self.experts),
                    experts_with_indexed_evidence=len(self.model.available),
                    method='compiled-classical-candidate-language-model',
                    execution='outward-score-intervals-and-exact-topk-replay',
                    exact_replay_candidates=selected['exact_candidates'],
                    ranking_semantics='Statistical topic relevance over supplied records; not a capability or industrial-suitability judgment.')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('-k',type=int,default=5)
    args = parser.parse_args()
    print(json.dumps(CompiledExpertSearch().search(args.query,args.k),ensure_ascii=False,indent=2))
