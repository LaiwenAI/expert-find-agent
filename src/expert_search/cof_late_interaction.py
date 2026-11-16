"""Frozen sentence/source late interaction: exploratory unit-of-aggregation control.

MaxSim-style aggregation is an existing technique, not a novelty claim. Query
sentences are not audited necessary conditions, and their scores are not truth.
"""
import hashlib
import importlib.metadata
import json
import re
import statistics
import time
import shutil
from pathlib import Path
import numpy as np
from .cof_benchmark import load, precision_metrics


def query_units(title, abstract):
    # CoF texts are lowercased: uppercase-only boundaries collapse the abstract.
    sentences = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Za-z])', abstract.strip())
                 if len(re.findall(r'[A-Za-z0-9]+', s)) >= 5]
    return list(dict.fromkeys(s.strip() for s in [title]+sentences if s.strip())) or [abstract.strip() or '[empty query]']


def aggregate(source_scores, members):
    """Compare max-min within a source to min-max within the same expert."""
    values = np.asarray(source_scores)[list(members)]
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Require finite source-by-query-unit matrix and nonempty membership')
    best = values.max(axis=0)
    return dict(expert_mean=float(best.mean()), expert_min=float(best.min()),
                single_source_min=float(values.min(axis=1).max()))


def run(dataset='SIGIR', model_name='BAAI/bge-small-en-v1.5'):
    from fastembed import TextEmbedding
    root, papers, reviewers, labels = load(dataset)
    out = root/'late-interaction-v2'
    out.mkdir(parents=True, exist_ok=True)
    source_ids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    texts = {pid: (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()[:1200] for pid in source_ids}
    unique = sorted(set(texts.values()))
    positions = {text: i for i, text in enumerate(unique)}
    memberships = {eid: sorted({positions[texts[pid]] for pid in r['papers']}) for eid,r in reviewers.items()}
    protocol = dict(dataset=dataset, model=model_name, fastembed=importlib.metadata.version('fastembed'),
        threads=1, window_size=1200, query_units='Title plus regex-split abstract segments of >=5 alphanumeric tokens; uppercase or lowercase boundary, duplicate strings removed; no LLM or qrels',
        source_sha256=hashlib.sha256(json.dumps([texts, memberships], sort_keys=True).encode()).hexdigest(),
        query_sha256=hashlib.sha256(json.dumps({qid: papers[qid] for qid in sorted(labels['soft'])}, sort_keys=True).encode()).hexdigest(),
        methods=['whole_max','expert_mean','expert_min','single_source_min'],
        warning='Prespecified fixed aggregations, no tuning. Query sentences are not logical requirements. Min controls may overconstrain. Source windows are literal prefixes, not abstract completeness certificates. Same released judged candidate sets; no absent-label negatives.')
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed late-interaction protocol')
    marker.write_text(json.dumps(protocol, indent=2))
    start = time.perf_counter()
    encoder = TextEmbedding(model_name=model_name, cache_dir='artifacts/models', threads=1)
    load_seconds = time.perf_counter()-start
    vectors_path = out/'vectors.npy'
    prior = root/'late-interaction-v1'
    if not vectors_path.exists() and (prior/'summary.json').exists():
        previous = json.loads((prior/'protocol.json').read_text())
        if all(previous[key] == protocol[key] for key in ('model','fastembed','threads','window_size','source_sha256')):
            recorded_hash = json.loads((prior/'summary.json').read_text())['vector_sha256']
            if hashlib.sha256((prior/'vectors.npy').read_bytes()).hexdigest() != recorded_hash:
                raise ValueError('Corrupt previous source index')
            shutil.copyfile(prior/'vectors.npy', vectors_path)
            shutil.copytree(prior/'parts', out/'parts', dirs_exist_ok=True)
            (out/'index-reuse.json').write_text(json.dumps(dict(source=str(prior), sha256=recorded_hash)))
    if not vectors_path.exists():
        parts = out/'parts'
        parts.mkdir(exist_ok=True)
        blocks = []
        for offset in range(0, len(unique), 256):
            path = parts/f'{offset}.npy'
            if not path.exists():
                start = time.perf_counter()
                block = np.asarray(list(encoder.passage_embed(unique[offset:offset+256], batch_size=32)), dtype=np.float32)
                block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-30)
                partial = path.with_suffix('.partial.npy')
                np.save(partial, block)
                partial.replace(path)
                (parts/f'{offset}.json').write_text(json.dumps(dict(encoding_seconds=time.perf_counter()-start, rows=len(block))))
            block = np.load(path)
            if len(block) != len(unique[offset:offset+256]) or not np.isfinite(block).all():
                raise ValueError('Invalid vector checkpoint')
            blocks.append(block)
            print(dataset, 'encoded', min(offset+256, len(unique)), '/', len(unique), flush=True)
        matrix = np.concatenate(blocks)
        partial = vectors_path.with_suffix('.partial.npy')
        np.save(partial, matrix)
        partial.replace(vectors_path)
    matrix = np.load(vectors_path)
    if len(matrix) != len(unique) or not np.isfinite(matrix).all():
        raise ValueError('Invalid source index')
    rows = []
    for qid in sorted(labels['soft']):
        target = out/f'query-{qid}.json'
        if target.exists():
            rows.append(json.loads(target.read_text()))
            continue
        if set(labels['soft'][qid]) != set(labels['hard'][qid]):
            raise ValueError('Different judged candidate sets')
        eligible = sorted(labels['soft'][qid])
        if any(not memberships[eid] for eid in eligible):
            raise ValueError('Judged expert has no sources')
        paper = papers[qid]
        start = time.perf_counter()
        units = query_units(paper.get('title',''), paper.get('abstract',''))
        segmentation = time.perf_counter()-start
        whole = (paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        start = time.perf_counter()
        query_vector = np.asarray(list(encoder.query_embed(whole))[0], dtype=np.float32)
        query_vector /= max(np.linalg.norm(query_vector), 1e-30)
        whole_encoding = time.perf_counter()-start
        start = time.perf_counter()
        whole_scores = matrix@query_vector
        scores = {'whole_max': {eid: float(whole_scores[memberships[eid]].max()) for eid in eligible}}
        whole_scoring = time.perf_counter()-start
        start = time.perf_counter()
        unit_vectors = np.asarray(list(encoder.query_embed(units, batch_size=32)), dtype=np.float32)
        unit_vectors /= np.maximum(np.linalg.norm(unit_vectors, axis=1, keepdims=True), 1e-30)
        unit_encoding = time.perf_counter()-start
        start = time.perf_counter()
        unit_scores = matrix@unit_vectors.T
        shared_scoring = time.perf_counter()-start
        aggregation_start = time.perf_counter()
        pooled = {eid: aggregate(unit_scores, memberships[eid]) for eid in eligible}
        combined_aggregation = time.perf_counter()-aggregation_start
        for method in ('expert_mean','expert_min','single_source_min'):
            scores[method] = {eid: values[method] for eid,values in pooled.items()}
        rankings, sorting = {}, {}
        for method, values in scores.items():
            start = time.perf_counter()
            rankings[method] = sorted(eligible, key=lambda eid: (-values[eid], eid))
            sorting[method] = time.perf_counter()-start
        row = dict(query_id=qid, query_units=len(units), rankings=rankings,
            metrics={task: {method: precision_metrics(rank, truth[qid]) for method,rank in rankings.items()} for task,truth in labels.items()},
            timing=dict(segmentation_seconds=segmentation, whole_encoding_seconds=whole_encoding,
                whole_scoring_seconds=whole_scoring, unit_encoding_seconds=unit_encoding,
                shared_unit_scoring_seconds=shared_scoring, combined_three_aggregations_seconds=combined_aggregation,
                sorting_seconds=sorting),
            whole_total_seconds=whole_encoding+whole_scoring+sorting['whole_max'],
            late_interaction_three_controls_seconds=segmentation+unit_encoding+shared_scoring+combined_aggregation+sum(sorting[m] for m in sorting if m!='whole_max'))
        target.write_text(json.dumps(row, indent=2))
        rows.append(row)
    report = dict(dataset=dataset, queries=len(rows), source_papers=len(source_ids), unique_source_windows=len(unique),
        vector_sha256=hashlib.sha256(vectors_path.read_bytes()).hexdigest(), model_load_seconds=load_seconds,
        cumulative_part_encoding_seconds=sum(json.loads(p.read_text())['encoding_seconds'] for p in (out/'parts').glob('*.json')),
        metrics={task: {method: {metric: statistics.mean(r['metrics'][task][method][metric] for r in rows)
                                for metric in rows[0]['metrics'][task][method]} for method in protocol['methods']} for task in labels},
        mean_query_units=statistics.mean(r['query_units'] for r in rows),
        median_whole_total_seconds=statistics.median(r['whole_total_seconds'] for r in rows),
        median_three_controls_seconds=statistics.median(r['late_interaction_three_controls_seconds'] for r in rows),
        warning=protocol['warning']+' Three-control timing includes all three pooling functions, not a claimed single-method latency. Warm persistent-model local query encoding included; cold load and corpus encoding separate. Unit splitting changes token allocation; whole-query encoding may truncate at the encoder limit. Concurrent network experiment is not a controlled hardware load test.')
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['SIGIR','NIPS'], default='SIGIR')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), indent=2))
