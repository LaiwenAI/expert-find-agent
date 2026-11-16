"""Paired exact ordered top-k timing on the full persisted teacher snapshot."""
import hashlib
import heapq
import json
import platform
import random
import statistics
import time
from pathlib import Path
from expert_search.compiled_expert_search import CompiledExpertSearch
from expert_search.corpus import records
from expert_search.certified_topk import certified_topk


def run():
    search=CompiledExpertSearch()
    model=search.model
    demands=[d for _,d in records('artifacts/demands/queries.jsonl')]
    root=Path('artifacts/certified-topk-v2');root.mkdir(parents=True,exist_ok=True)
    spec=dict(version=2,corpus_sha256=search.snapshot,
        queries_sha256=hashlib.sha256(Path('artifacts/demands/queries.jsonl').read_bytes()).hexdigest(),
        code_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            'src/expert_search/certified_topk.py','src/expert_search/compiled_language_model.py',
            'experiments/certified_topk_benchmark.py')},k=5,repetitions=5,seed=20261117,
        python=platform.python_version(),machine=platform.machine(),
        scope='Both methods return exact ordered top5 scores; full compiled-score plus heap selection versus outward score intervals plus exact replay. No retrieval quality gain claimed.')
    marker=root/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Changed benchmark protocol')
    marker.write_text(json.dumps(spec,indent=2))
    # Shared warm-up, excluded from paired measured repetitions.
    model.score(demands[0]['query']);certified_topk(model,demands[0]['query'],5)
    rng=random.Random(spec['seed']);trials=[]
    for demand in demands:
        for repetition in range(spec['repetitions']):
            order=['full','certified'];rng.shuffle(order)
            outputs={};timings={}
            for method in order:
                start=time.perf_counter()
                if method=='full':
                    scores=model.score(demand['query'])
                    selected=heapq.nsmallest(5,scores,key=lambda e:(-scores[e],e))
                    outputs[method]=dict(results=[dict(id=e,score=scores[e]) for e in selected])
                else:outputs[method]=certified_topk(model,demand['query'],5)
                timings[method]=time.perf_counter()-start
            if outputs['full']['results']!=outputs['certified']['results']:
                raise ValueError('Ordered top5 or exact scores changed')
            trials.append(dict(query_id=demand['id'],repetition=repetition,order=order,
                full_seconds=timings['full'],certified_seconds=timings['certified'],
                exact_candidates=outputs['certified']['exact_candidates'],
                total_candidates=outputs['certified']['total_candidates'],
                postings_visited=outputs['certified']['postings_visited']))
        print(demand['id'],'five paired repetitions complete',flush=True)
    report=dict(protocol=spec,queries=len(demands),trials=trials,exact_ordered_top5=True,
        initialization_seconds=search.initialization_seconds,
        mean_full_seconds=statistics.mean(t['full_seconds'] for t in trials),
        mean_certified_seconds=statistics.mean(t['certified_seconds'] for t in trials),
        median_full_seconds=statistics.median(t['full_seconds'] for t in trials),
        median_certified_seconds=statistics.median(t['certified_seconds'] for t in trials),
        mean_exact_candidates=statistics.mean(t['exact_candidates'] for t in trials),
        max_exact_candidates=max(t['exact_candidates'] for t in trials),
        warning='Interval pass still visits all active postings and maintains corpus-sized bounds. Exact replay candidate reduction is not posting or neural-call reduction. Warm local scoring and top5 selection only; no quality labels, online encoder, agent calls or end-to-end SLA. Repetitions are paired timing samples, not independent queries.')
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    Path('docs/exploration/teacher-certified-topk.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','protocol')},indent=2))


if __name__=='__main__':run()
