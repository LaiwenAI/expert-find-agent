"""Compare source-scoped max-min expert reranking at a fixed citation budget."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, sha
from experiments.retrieval.cof_query_encoding_budget import compare_labels
from experiments.retrieval.cof_segmented_gpu_ownership import prepare_cohort

SHORTLIST = 20
SOURCES = 6
COEFFICIENT = .5
ARMS = ('dense', 'fragment_mean', 'single_bottleneck', 'pair_bottleneck')
REPEATS = 3
SEED = 20261122
ENDS = re.compile(r'(?<=[.!?])\s+|;\s+')


def facets(text):
    title, _, abstract = text.partition('\n')
    fragments = [s.strip() for s in ENDS.split(abstract) if len(s.split()) >= 8]
    out = list(dict.fromkeys(t for t in (title.strip(), *fragments[:2]) if len(t.split()) >= 4))
    return out if len(out) >= 2 else []


def owned_candidates(ranker, vector):
    scores = ranker.score(vector).cpu().numpy()
    index = ranker.index
    maxima = np.maximum.reduceat(scores[index.owner_rows], index.owner_starts)
    shortlist = np.argsort(-maxima, kind='stable')[:SHORTLIST]
    pools = []
    for owner_index in shortlist:
        owner = index.owner_ids[owner_index]
        sources = np.asarray(index.owners[owner])
        order = np.argsort(-scores[sources], kind='stable')[:SOURCES]
        pools.append((owner, float(maxima[owner_index]), sources[order].tolist()))
    compact = np.asarray(sorted({source for _, _, sources in pools for source in sources}))
    return pools, compact


def local_choice(source_rows, values, method):
    if values is None:
        return 0.0, source_rows[:2]
    if method == 'fragment_mean':
        scores = values.mean(axis=1)
        order = np.argsort(-scores, kind='stable')
        return float(scores[order[0]]), [source_rows[i] for i in order[:2]]

    worst = values.min(axis=1)
    single = int(np.argmax(worst))
    alternative = next((i for i in range(len(values)) if i != single), None)
    if method == 'single_bottleneck' or alternative is None:
        keep = [source_rows[single]]
        if alternative is not None:
            keep.append(source_rows[alternative])
        return float(worst[single]), keep
    pairs = np.asarray(list(combinations(range(len(values)), 2)))
    joint = np.maximum(values[pairs[:, 0]], values[pairs[:, 1]]).min(axis=1)
    best = int(np.argmax(joint))
    return float(joint[best]), [source_rows[i] for i in pairs[best]]


def select(pools, compact, matrix, method):
    rows = []
    mapping = {source: i for i, source in enumerate(compact)} if matrix is not None else {}
    for owner, anchor, sources in pools:
        values = None if matrix is None else matrix[[mapping[i] for i in sources]]
        extra, witnesses = local_choice(sources, values, method)
        rows.append((owner, anchor + COEFFICIENT * extra, witnesses))
    rows.sort(key=lambda r: -r[1])
    return rows


def deliver(ranking, ranker, conn):
    ids = [[ranker.index.source_ids[source] for source in sources] for _, _, sources in ranking]
    all_ids = sorted(set().union(*map(set, ids)))
    placeholders = ','.join('?' for _ in all_ids)
    excerpts = dict(conn.execute(f'SELECT id, excerpt FROM source WHERE id IN ({placeholders})', all_ids))
    body = []
    chars = 0
    for (owner, score, _), selected in zip(ranking, ids, strict=True):
        text = [(source, excerpts[source]) for source in selected]
        n = sum(len(excerpt) for _, excerpt in text)
        if len(text) > 2 or n > 1000:
            raise ValueError('Evidence budget exceeded')
        chars += n
        body.append({'expert': owner, 'score': score, 'citations': text})
    return json.dumps(body, ensure_ascii=False).encode(), ids, chars


def dataset_run(name, out, checkpoint, encoder):
    ranker, db, queries, qrels, inventory, _ = prepare_cohort(name, out, checkpoint)
    encoder.batch_size = 4
    generator = random.Random(SEED)
    rows = []
    try:
        with sqlite3.connect(db) as conn:
            for q, text in sorted(queries.items()):
                started = perf_counter()
                fragments = facets(text)
                split_ms = (perf_counter() - started) * 1000
                started = perf_counter()
                pools, compact = owned_candidates(ranker, encoder.encode([text], query=True)[0])
                base_ms = (perf_counter() - started) * 1000
                if fragments:
                    started = perf_counter()
                    representations = encoder.encode(fragments, query=True)
                    device_queries = torch.as_tensor(representations, device='cuda')
                    docs = ranker.vectors.index_select(0, torch.as_tensor(compact, device='cuda'))
                    matrix = (docs @ device_queries.T).cpu().numpy()
                    extra_ms = split_ms + (perf_counter() - started) * 1000
                else:
                    matrix, extra_ms = None, split_ms
                by_method = {}
                for _ in range(REPEATS):
                    order = list(ARMS)
                    generator.shuffle(order)
                    for method in order:
                        started = perf_counter()
                        ranking = select(pools, compact, None if method == 'dense' else matrix, method)
                        raw, citations, characters = deliver(ranking[:TOP_K], ranker, conn)
                        elapsed = (perf_counter() - started) * 1000
                        experts = [e for e, _, _ in ranking]
                        entry = by_method.setdefault(method, {'ranking': experts, 'citations': citations,
                                                             'characters': characters, 'ms': []})
                        if entry['ranking'] != experts or entry['citations'] != citations or not raw:
                            raise ValueError('Unstable retrieval')
                        entry['ms'].append(base_ms + (extra_ms if method != 'dense' else 0.0) + elapsed)
                rows.append({'query': q, 'fragments': len(fragments), 'candidate_sources': len(compact),
                             'arms': by_method})
    finally:
        db.unlink()
    arms = {}
    for method in ARMS:
        observed = [r['arms'][method] for r in rows]
        report = {'p95_ms': float(np.quantile([v for r in observed for v in r['ms']], .95)),
                  'mean_chars': mean(r['characters'] for r in observed),
                  'same_experts_and_citations': sum(r['arms']['dense']['ranking'][:TOP_K] == a['ranking'][:TOP_K]
                                                  and r['arms']['dense']['citations'] == a['citations']
                                                  for r, a in zip(rows, observed, strict=True)),
                  'full': {}, 'judged_within_shortlist': {}}
        for grade, labels in qrels.items():
            full, judged = [], []
            for row, a in zip(rows, observed, strict=True):
                local = labels[row['query']]
                base = row['arms']['dense']['ranking']
                full.append(compare_labels(base[:TOP_K], a['ranking'][:TOP_K], local))
                filtered_base = [e for e in base if e in local][:TOP_K]
                if len(filtered_base) == TOP_K:
                    filtered_ranking = [e for e in a['ranking'] if e in local][:TOP_K]
                    judged.append(compare_labels(filtered_base, filtered_ranking, local))
            for k, records in (('full', full), ('judged_within_shortlist', judged)):
                report[k][grade] = {'queries': len(records)}
                if records:
                    report[k][grade].update({m: mean(r[m] for r in records) for m in records[0]})
                    report[k][grade]['paired_known_delta'] = paired_interval(
                        [r['known_difference'] for r in records], repeats=4000)
        arms[method] = report
    pair_vs_single = {}
    for grade, labels in qrels.items():
        delta = [compare_labels(r['arms']['single_bottleneck']['ranking'][:TOP_K],
                                r['arms']['pair_bottleneck']['ranking'][:TOP_K], labels[r['query']])
                 for r in rows]
        pair_vs_single[grade] = {k: mean(r[k] for r in delta)
                                 for k in ('difference_lower', 'difference_upper', 'known_difference')}
        pair_vs_single[grade]['paired_known_delta'] = paired_interval(
            [r['known_difference'] for r in delta], repeats=4000)
    target = out / f'{name}-rows.json'
    target.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    return {'index': inventory, 'rows_sha256': sha(target), 'multi_fragment': sum(r['fragments'] >= 2 for r in rows),
            'mean_candidate_sources': mean(r['candidate_sources'] for r in rows), 'arms': arms,
            'pair_minus_single': pair_vs_single}


def run(out):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm GPU allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed encoder checkpoint')
    out.mkdir(parents=True, exist_ok=False)
    encoder = QwenTorchEmbedding(batch_size=4)
    encoder.model.config.use_cache = False
    encoder.encode(['warm up', 'fixed scoring', 'label scope'], query=True)
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'code_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(checkpoint),
        'datasets': {d: dataset_run(d, out, checkpoint, encoder) for d in ('SIGIR', 'SciRepEval')},
        'scope': 'Reused expert-ranking cohorts with exact profile-overlap exclusions. All methods rerank '
                 'the same initial 20 experts and use at most two owned 500-character excerpts per expert. '
                 'Fragment similarities use frozen cached full-title-and-abstract source embeddings. Heuristic '
                 'subqueries are not verified atomic requirements. Reviewer relevance does not grade citations. '
                 'The judged-only subset is a diagnostic within the retrieved shortlist. Timing includes '
                 'shared per-query encoding, exact retrieval, additional facet encoding, rank and SQLite '
                 'excerpt delivery, excluding semantic verification. Shared encoding time is measured once.',
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
