"""Fuse exact multi-owner witness reduction, and compare complete citation responses."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
from time import perf_counter

import numpy as np
import torch
import triton
import triton.language as tl

from expert_search.cof_benchmark import load
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, serialize_hits, sha

METHODS = ('host', 'scatter', 'segmented')
REPEATS = 5
SEED = 20261119
TILE = 128


@triton.jit
def segmented_maximum(scores, edge_sources, offsets, maxima, witnesses, TILE: tl.constexpr):
    owner = tl.program_id(0)
    begin = tl.load(offsets + owner)
    end = tl.load(offsets + owner + 1)
    lanes = tl.arange(0, TILE)
    best_score = tl.full((), float('-inf'), tl.float32)
    best_source = tl.full((), 2147483647, tl.int32)
    for tile in range(tl.cdiv(end - begin, TILE)):
        edges = begin + tile * TILE + lanes
        valid = edges < end
        sources = tl.load(edge_sources + edges, valid, other=0)
        values = tl.load(scores + sources, valid, other=float('-inf'))
        score = tl.max(values, 0)
        witness = tl.min(tl.where(valid & (values == score), sources, 2147483647), 0)
        best_source = tl.where(score > best_score, witness,
                               tl.where(score == best_score, tl.minimum(witness, best_source), best_source))
        best_score = tl.maximum(best_score, score)
    tl.store(maxima + owner, tl.load(scores + best_source))
    tl.store(witnesses + owner, best_source)


@triton.jit
def pack_top(maxima, witnesses, packed, OWNERS: tl.constexpr, WIDTH: tl.constexpr, K: tl.constexpr):
    owners = tl.arange(0, WIDTH)
    remaining = tl.load(maxima + owners, owners < OWNERS, other=float('-inf'))
    for rank in tl.static_range(K):
        score = tl.max(remaining, 0)
        owner = tl.min(tl.where(remaining == score, owners, 2147483647), 0)
        source = tl.load(witnesses + owner)
        tl.store(packed + rank * 3, owner)
        tl.store(packed + rank * 3 + 1, source)
        tl.store(packed + rank * 3 + 2, score.to(tl.int32, bitcast=True))
        remaining = tl.where(owners == owner, float('-inf'), remaining)


class SegmentedOwnerRanking:
    """CSR owner-to-source adjacency with explicit stable expert and source ties."""

    def __init__(self, ranker):
        index = ranker.index
        count = len(index.owner_ids)
        if not TOP_K <= count <= 8192 or len(index.owner_rows) >= np.iinfo(np.int32).max:
            raise ValueError('Unsupported registry size for this experimental kernel')
        self.rows = ranker.owner_rows.to(torch.int32)
        self.offsets = torch.tensor([*index.owner_starts, len(index.owner_rows)], device='cuda', dtype=torch.int32)
        self.maxima = torch.empty(count, device='cuda', dtype=torch.float32)
        self.witnesses = torch.empty(count, device='cuda', dtype=torch.int32)
        self.packed = torch.empty((TOP_K, 3), device='cuda', dtype=torch.int32)

    def rank(self, scores):
        n = self.maxima.numel()
        segmented_maximum[(n,)](scores, self.rows, self.offsets, self.maxima, self.witnesses, TILE, num_warps=4)
        pack_top[(1,)](self.maxima, self.witnesses, self.packed, n, triton.next_power_of_2(n), TOP_K, num_warps=4)
        output = self.packed.cpu().numpy()
        return list(zip(output[:, 0].tolist(), output[:, 1].tolist(),
                        output[:, 2].view(np.float32).tolist(), strict=True))

    @property
    def bytes(self):
        return sum(a.numel() * a.element_size() for a in
                   (self.rows, self.offsets, self.maxima, self.witnesses, self.packed))


def run_dataset(dataset, folder, checkpoint, encoder):
    root, papers, profiles, grades = load(dataset)
    meta = json.loads((root / 'manifest.json').read_text())
    omit = set(meta['exact_query_texts_in_profiles']) | set(meta['query_ids_in_profiles'])
    source_ids = {p for profile in profiles.values() for p in profile['papers']}
    source_titles = {' '.join(papers[p]['title'].casefold().split()) for p in source_ids}
    omit.update(q for q in grades['soft'] if ' '.join(papers[q]['title'].casefold().split()) in source_titles)
    ranker, db, queries, labels, inventory = prepare(dataset, folder, checkpoint, exclude_queries=omit)
    device = SegmentedOwnerRanking(ranker)
    ranks = {'host': ranker.rank_host, 'scatter': ranker.rank_device, 'segmented': device.rank}

    rng = random.Random(SEED)
    cases = sorted(queries)
    # Exact ties, negative scores, and many repeated scores including shared sources.
    for values in (torch.full((len(source_ids),), -.5, device='cuda'),
                   torch.arange(len(source_ids), device='cuda').remainder(7).float() - 3):
        expected = ranks['host'](values)
        assert all(rank(values) == expected for rank in ranks.values())
    print(dataset, 'exact tie controls passed', len(cases), flush=True)

    def request(qid, rank, vector=None):
        start = perf_counter()
        if vector is None:
            vector = encoder.encode([queries[qid]], query=True)[0]
        before = perf_counter()
        ranked = rank(ranker.score(vector))
        retrieved = perf_counter()
        response, experts = serialize_hits(ranker, connection, ranked)
        end = perf_counter()
        return response, experts, [(retrieved - before) * 1000, (end - start) * 1000]

    rows = {q: {'online': {m: [] for m in METHODS}, 'cached': {m: [] for m in METHODS}} for q in cases}
    try:
        with sqlite3.connect(db) as connection:
            for m in METHODS:
                request(cases[0], ranks[m])
            vectors = {q: encoder.encode([queries[q]], query=True)[0] for q in cases}
            for repeat in range(REPEATS):
                rng.shuffle(cases)
                for i, q in enumerate(cases):
                    order = (i + repeat) % len(METHODS)
                    for method in METHODS[order:] + METHODS[:order]:
                        for mode in ('online', 'cached'):
                            response, experts, timing = request(q, ranks[method], vectors[q] if mode == 'cached' else None)
                            digest = hashlib.sha256(response).hexdigest()
                            row = rows[q]
                            if row.setdefault('response_sha256', digest) != digest:
                                raise ValueError(f'Exact responses differ for {dataset}/{q}/{method}/{mode}')
                            row.setdefault('experts', experts)
                            row[mode][method].append(timing)
    finally:
        db.unlink()
    path = folder / f'{dataset}-rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    means = {}
    for mode in ('online', 'cached'):
        results = {}
        for method in METHODS:
            values = np.array([sample for row in rows.values() for sample in row[mode][method]])
            results[method] = {
                'p50_search_ms': float(np.median(values[:, 0])),
                'p95_search_ms': float(np.quantile(values[:, 0], .95)),
                'p50_total_ms': float(np.median(values[:, 1])),
                'p95_total_ms': float(np.quantile(values[:, 1], .95)),
            }
        means[mode] = results
    changes = np.array([np.mean(row['online']['segmented'], axis=0) - np.mean(row['online']['host'], axis=0)
                        for row in rows.values()])
    graded = {name: {
        'known_positive_at_5': sum(by_query[q].get(e) == 1 for q, r in rows.items() for e in r['experts']) / (TOP_K * len(rows)),
        'judged_coverage_at_5': sum(e in by_query[q] for q, r in rows.items() for e in r['experts']) / (TOP_K * len(rows)),
    } for name, by_query in labels.items()}
    ownership = Counter(p for profile in profiles.values() for p in set(profile['papers']))
    return {'inventory': inventory, 'max_owner_sources': max(map(len, ranker.index.owners.values())),
            'sources_with_multiple_owners': sum(n > 1 for n in ownership.values()),
            'segmented_device_metadata_and_scratch_bytes': device.bytes,
            'rows_sha256': sha(path), 'methods': means, 'shared_quality': graded,
            'paired_segmented_minus_host_ms': {k: paired_interval(changes[:, i], repeats=4000)
                                               for i, k in enumerate(('search', 'total'))}}


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for file, digest in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / file) != digest:
            raise ValueError('Changed embedding checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    encoder = QwenTorchEmbedding(batch_size=1)
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'driver_stack': {'torch': torch.__version__, 'triton': triton.__version__},
        'code_sha256': sha(Path(__file__)), 'baseline_code_sha256': sha(Path(__file__).with_name('cof_online_ownership.py')),
        'model_sha256': sha(checkpoint), 'repeats': REPEATS,
        'datasets': {d: run_dataset(d, output, checkpoint, encoder) for d in ('SIGIR', 'SciRepEval')},
        'scope': 'Prewarmed sequential requests with exact scoring, rank aggregation and SQLite citation delivery. '
                 'Bytewise comparison includes scores and citations, with no partial-label imputation. '
                 'No service queue, network, semantic verification or GPU kernel JIT in reported timing.',
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
