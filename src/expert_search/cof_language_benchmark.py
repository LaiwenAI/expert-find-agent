"""Classical expert-finding baselines on unchanged CoF candidate/label protocols."""
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
from .cof_benchmark import load, precision_metrics
from .expert_language_models import ExpertLanguageModels
from .retrieval import tokens


def run(dataset='SIGIR'):
    root, papers, reviewers, labels = load(dataset)
    out = root/'language-models-v2'
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    documents = {pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in ids}
    profiles = {eid:r['papers'] for eid,r in reviewers.items()}
    protocol = dict(dataset=dataset, smoothing=.5, scored_candidates='All profiles with nonempty text; rank all before judgment-only evaluation filtering', association='Binary paper-ID ownership; document-centric 1/number of nonempty profile papers; candidate-centric 1/number of linked experts',
        source_sha256=hashlib.sha256(json.dumps([documents,profiles], sort_keys=True).encode()).hexdigest(),
        query_sha256=hashlib.sha256(json.dumps({qid:papers[qid] for qid in labels['soft']}, sort_keys=True).encode()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(inspect.getsource(tokens).encode()).hexdigest(),
        model_source_sha256=hashlib.sha256(inspect.getsource(ExpertLanguageModels).encode()).hexdigest(),
        reference='https://doi.org/10.1145/1148170.1148181',
        warning='Classical baselines, not new methods or an exact reproduction of TREC numbers. Fixed lambda=.5, not selected on CoF labels. No stemming or stopword filtering. Empty text records omitted with association weights renormalized; no inference of absent expertise. Outside-vocabulary query tokens ignored. Original candidate sets and source IDs retained; no exact-text merge.')
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed language-model protocol')
    marker.write_text(json.dumps(protocol, indent=2))
    start = time.perf_counter()
    index = ExpertLanguageModels(documents, profiles)
    build_seconds = time.perf_counter()-start
    rows = []
    for qid in sorted(labels['soft']):
        target = out/f'query-{qid}.json'
        if target.exists():
            rows.append(json.loads(target.read_text()))
            continue
        if set(labels['soft'][qid]) != set(labels['hard'][qid]):
            raise ValueError('Different judged candidate sets')
        query = (papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        rankings, full_rankings, seconds = {}, {}, {}
        usable = [eid for eid in index.experts if index.members[eid]]
        if set(labels['soft'][qid])-set(usable):
            raise ValueError('Judged candidate has no usable source')
        for mode in index.MODES:
            start = time.perf_counter()
            scores = index.score(query, mode, usable)
            full_rankings[mode] = sorted(scores, key=lambda eid:(-scores[eid],eid))
            seconds[mode] = time.perf_counter()-start
            rankings[mode] = [eid for eid in full_rankings[mode] if eid in labels['soft'][qid]]
        row = dict(query_id=qid, rankings=rankings, full_rankings=full_rankings, seconds=seconds,
                   metrics={task:{mode:precision_metrics(rank,truth[qid]) for mode,rank in rankings.items()} for task,truth in labels.items()})
        target.write_text(json.dumps(row, indent=2))
        rows.append(row)
    report = dict(dataset=dataset, queries=len(rows), empty_source_records=len(index.empty_sources),
        initialization_seconds=build_seconds,
        metrics={task:{mode:{key:statistics.mean(r['metrics'][task][mode][key] for r in rows)
                            for key in rows[0]['metrics'][task][mode]} for mode in index.MODES} for task in labels},
        median_query_seconds={mode:statistics.median(r['seconds'][mode] for r in rows) for mode in index.MODES},
        warning=protocol['warning']+' Query time includes tokenization, scoring ALL nonempty profiles and sorting; initialization is separate. Quality is evaluated only after filtering to released judged candidates, never a claim of full-catalog relevance. Single sequential pass, not a repeated controlled latency comparison.')
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['SIGIR','NIPS'], default='SIGIR')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), indent=2))
