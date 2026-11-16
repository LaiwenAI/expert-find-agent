"""Paired repeated local timing, exact-score audit and unchanged-label quality."""
import hashlib
import inspect
import json
import random
import statistics
import time
from pathlib import Path
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.fast_retrieval import FastSearch


def run(dataset):
    root,papers,reviewers,labels = load(dataset)
    ids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    documents = {pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in ids}
    start = time.perf_counter()
    original = ExpertLanguageModels(documents,{eid:r['papers'] for eid,r in reviewers.items()})
    reference_build = time.perf_counter()-start
    start = time.perf_counter()
    compiled = CompiledCandidateLM(original)
    compilation = time.perf_counter()-start
    available = sorted(compiled.available)
    sparse = FastSearch(root/'unique-profile-corpus',root/'unique-profile-corpus/fast.sqlite')
    def lexical(query):
        values = sparse.document_scores(query)
        return {eid:max((values.get(pid,0.) for pid in set(reviewers[eid]['papers'])),default=0.) for eid in available}
    methods = {'reference-model1':lambda q:original.score(q,eligible=available),
               'compiled-model1':lambda q:compiled.score(q,available), 'bm25-max':lexical}
    trials,rankings = [],{}
    rng = random.Random(20261116)
    for qid in sorted(labels['soft']):
        query = (papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        previous = None
        for repetition in range(3):
            order = list(methods)
            rng.shuffle(order)
            scored = {}
            for method in order:
                start = time.perf_counter()
                scores = methods[method](query)
                rank = sorted(scores,key=lambda eid:(-scores[eid],eid))
                elapsed = time.perf_counter()-start
                scored[method] = scores
                trials.append(dict(query_id=qid,repetition=repetition,method=method,seconds=elapsed))
                filtered = [eid for eid in rank if eid in labels['soft'][qid]]
                rankings.setdefault(qid,{})[method] = filtered
            if scored['reference-model1'] != scored['compiled-model1']:
                raise ValueError('Compiled scores differ from reference')
            if previous is not None and previous != scored:
                raise ValueError('Repeated scores are unstable')
            previous = scored
    report = dict(dataset=dataset,queries=len(rankings),repetitions=3,seed=20261116,
        scored_profiles=len(available), reference_build_seconds=reference_build,compilation_seconds=compilation,
        compiled_array_bytes=compiled.array_bytes,compiled_nonzero_postings=compiled.nonzero_postings,
        metrics={task:{method:{key:statistics.mean(precision_metrics(rankings[qid][method],truth[qid])[key] for qid in rankings)
                              for key in ('p@5','p@10','ndcg@10')} for method in methods} for task,truth in labels.items()},
        timings={method:dict(mean=statistics.mean(t['seconds'] for t in trials if t['method']==method),
                            median=statistics.median(t['seconds'] for t in trials if t['method']==method)) for method in methods},
        trials=trials,exact_scores_match=True,
        source_sha256=hashlib.sha256(json.dumps([documents,reviewers],sort_keys=True).encode()).hexdigest(),
        compiler_sha256=hashlib.sha256(inspect.getsource(CompiledCandidateLM).encode()).hexdigest(),
        warning='Engineering compilation of a classical language model, not a novel ranker or model training. All usable profiles scored and sorted; quality filters only to released judged candidates. Query tokenization included, offline construction separate. Repeated paired method order randomized per query. Concurrent API worker exists; not isolated hardware or an end-to-end agent SLA. Three repetitions of the same queries are not independent quality samples.')
    destination = Path('docs/exploration')/f'compiled-lm-{dataset.lower()}.json'
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(output=str(destination),timings=report['timings'],metrics=report['metrics']),indent=2))


if __name__=='__main__':
    # Deliberately sequential: these jobs measure shared-CPU runtime.
    run('SIGIR')
    run('NIPS')
