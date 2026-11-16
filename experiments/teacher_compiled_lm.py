"""Full teacher-corpus index build/load and exact-score timing, with no quality claim."""
import hashlib
import json
import statistics
import time
import random
from collections import defaultdict
from pathlib import Path
from expert_search.corpus import records
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM


def run():
    evidence = Path('artifacts/corpus/evidence.jsonl')
    experts = Path('artifacts/corpus/experts.jsonl')
    snapshot = hashlib.sha256((hashlib.sha256(evidence.read_bytes()).hexdigest()+hashlib.sha256(experts.read_bytes()).hexdigest()).encode()).hexdigest()
    documents,profiles = {},defaultdict(list)
    for _,row in records(evidence):
        if row['id'] in documents:
            raise ValueError('Duplicate evidence ID')
        documents[row['id']] = row['text']
        profiles[row['expert_id']].append(row['id'])
    start = time.perf_counter()
    reference = ExpertLanguageModels(documents,profiles)
    build = time.perf_counter()-start
    print('reference built',round(build,3),flush=True)
    start = time.perf_counter()
    compiled = CompiledCandidateLM(reference)
    compilation = time.perf_counter()-start
    target = Path('artifacts/teacher-compiled-lm')
    start = time.perf_counter()
    compiled.save(target,snapshot)
    save = time.perf_counter()-start
    start = time.perf_counter()
    loaded = CompiledCandidateLM.load(target,snapshot)
    loading = time.perf_counter()-start
    print('compiled/saved/loaded',round(compilation,3),round(save,3),round(loading,3),flush=True)
    trials = []
    available = sorted(loaded.available)
    rng = random.Random(20261116)
    demands = [demand for _,demand in records('artifacts/demands/queries.jsonl')]
    for demand in demands:
        for repetition in range(3):
            order = ['reference','compiled']
            rng.shuffle(order)
            scores,ranks,seconds = {},{},{}
            for method in order:
                start = time.perf_counter()
                scores[method] = (reference.score(demand['query'],eligible=available) if method=='reference'
                                  else loaded.score(demand['query'],available))
                ranks[method] = sorted(scores[method],key=lambda eid:(-scores[method][eid],eid))
                seconds[method] = time.perf_counter()-start
            if scores['compiled'] != scores['reference'] or ranks['compiled'] != ranks['reference']:
                raise ValueError('Compiled full-corpus scores or ranks differ')
            trials.append(dict(query_id=demand['id'],repetition=repetition,order=order,
                               reference_seconds=seconds['reference'],compiled_seconds=seconds['compiled']))
        print(demand['id'],'three paired repetitions complete',flush=True)
    report = dict(experts_total=sum(1 for _ in records(experts)),scored_profiles=len(available),
        evidence_records=len(documents),empty_sources=len(reference.empty_sources),
        queries=len(demands),repetitions=3,seed=20261116,source_sha256=snapshot,array_bytes=loaded.array_bytes,
        index_disk_bytes=sum(p.stat().st_size for p in target.iterdir() if p.is_file()),
        reference_build_seconds=build,compilation_seconds=compilation,save_seconds=save,load_seconds=loading,
        mean_reference_seconds=statistics.mean(r['reference_seconds'] for r in trials),
        mean_compiled_seconds=statistics.mean(r['compiled_seconds'] for r in trials),trials=trials,
        median_reference_seconds=statistics.median(r['reference_seconds'] for r in trials),
        median_compiled_seconds=statistics.median(r['compiled_seconds'] for r in trials),
        exact_scores_and_ranks=True,
        warning='Real teacher corpus and original 20 full demand queries; no relevance judgments or quality improvement claim. Exact compilation of classical LM only. Three paired repetitions with randomized method order per query; repetitions are not new independent quality samples. Full-corpus score and sort included; offline normalization, compilation, serialization and loading separate. Other experimental workers may use CPU; not an isolated end-to-end service SLA. Arrays exclude Python/object/text memory.')
    Path('docs/exploration/teacher-compiled-lm.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)


if __name__=='__main__':
    run()
