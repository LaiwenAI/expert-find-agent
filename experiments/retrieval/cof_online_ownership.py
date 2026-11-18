"""Time exact reviewer ranking and citation delivery with device-side ownership reduction."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sqlite3
from statistics import mean
import time

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval

EMBEDDING = Path('artifacts/models/qwen3-embedding-0.6b')
METHODS = ('host_reduction', 'device_reduction')
TOP_K = 5
REPEATS = 7
SEED = 20261118


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text(paper):
    return (paper.get('title', '') + '\n' + paper.get('abstract', '')).strip()


class OwnerRanking:
    """Identical maximum-score and lexicographic witness rules on either device."""

    def __init__(self, index):
        if max(len(index.owner_ids), len(index.source_ids)) >= np.iinfo(np.int32).max:
            raise ValueError('Result identifiers must fit the packed int32 response')
        self.index = index
        self.vectors = torch.tensor(index.vectors, device='cuda')
        starts = index.owner_starts
        counts = np.diff(np.append(starts, len(index.owner_rows)))
        self.owner_rows = torch.tensor(index.owner_rows, device='cuda')
        self.owner_positions = torch.tensor(np.repeat(np.arange(len(counts)), counts), device='cuda')
        torch.cuda.synchronize()

    def score(self, query):
        with torch.inference_mode():
            return torch.mv(self.vectors, torch.as_tensor(query, device='cuda'))

    def rank_host(self, scores):
        scores = scores.cpu().numpy()
        maxima = np.maximum.reduceat(scores[self.index.owner_rows], self.index.owner_starts)
        best = np.argsort(-maxima, kind='stable')[:TOP_K]
        results = []
        for owner in best:
            rows = self.index.owners[self.index.owner_ids[owner]]
            witness = rows[int(np.argmax(scores[rows]))]
            results.append((int(owner), int(witness), float(maxima[owner])))
        return results

    def rank_device(self, scores):
        owner_count = len(self.index.owner_ids)
        source_count = len(self.index.source_ids)
        edge_scores = scores.index_select(0, self.owner_rows)
        maxima = scores.new_full((owner_count,), float('-inf'))
        maxima.scatter_reduce_(0, self.owner_positions, edge_scores, reduce='amax')
        matched = edge_scores == maxima.index_select(0, self.owner_positions)
        valid_rows = torch.where(matched, self.owner_rows, source_count)
        witnesses = self.owner_rows.new_full((owner_count,), source_count)
        witnesses.scatter_reduce_(0, self.owner_positions, valid_rows, reduce='amin')
        best = torch.argsort(maxima, descending=True, stable=True)[:TOP_K]
        # Move one compact buffer to the host; reinterpret the third int32 as its FP32 bits.
        packed = torch.stack((
            best.to(torch.int32), witnesses[best].to(torch.int32), maxima[best].view(torch.int32),
        ), dim=1).cpu().numpy()
        return list(zip(
            packed[:, 0].tolist(), packed[:, 1].tolist(),
            packed[:, 2].view(np.float32).tolist(), strict=True,
        ))


def prepare(dataset, folder, model_manifest):
    root, papers, reviewers, labels = load(dataset)
    ownership = defaultdict(list)
    for reviewer, profile in reviewers.items():
        for paper in set(profile['papers']):
            ownership[paper].append(reviewer)
    sources = sorted(ownership)
    queries = sorted(labels['soft'])
    inventory = json.loads((root / 'manifest.json').read_text())
    if inventory['exact_query_texts_in_profiles'] or inventory['query_ids_in_profiles']:
        raise ValueError('Queries present in profiles')
    vector_dir = root / 'qwen-embedding-torch-v1'
    protocol = json.loads((vector_dir / 'protocol.json').read_text())
    summary = json.loads((vector_dir / 'summary.json').read_text())
    vectors = vector_dir / 'vectors.npy'
    if sha(vectors) != summary['vectors_sha256']:
        raise ValueError('Changed source vectors')
    if protocol['checkpoint'] != json.loads(model_manifest.read_text()):
        raise ValueError('Embedding checkpoint differs from index')
    items = [dict(id='doc:' + key, text=text(papers[key]), query=False) for key in sources]
    items += [dict(id='query:' + key, text=text(papers[key]), query=True) for key in queries]
    if hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest() != protocol['input_sha256']:
        raise ValueError('Changed embedding input order')
    matrix = np.load(vectors, allow_pickle=False)
    if matrix.shape != (len(items), 1024) or matrix.dtype != np.float32:
        raise ValueError('Unexpected vector matrix')
    started = time.perf_counter()
    db = folder / f'{dataset}-sources.sqlite'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT, excerpt TEXT)')
        connection.executemany('INSERT INTO source VALUES (?, ?, ?)', (
            (source, papers[source].get('title', ''), text(papers[source])[:500])
            for source in sources
        ))
    storage_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index = ExactOwnedVectorIndex(matrix[:len(sources)], sources, [ownership[s] for s in sources])
    ranker = OwnerRanking(index)
    index_seconds = time.perf_counter() - started
    metadata = {
        'queries': len(queries), 'sources': len(sources), 'owners': len(index.owners),
        'ownership_edges': len(index.owner_rows), 'vectors_sha256': sha(vectors),
        'source_store_build_seconds': storage_seconds,
        'source_store_bytes': db.stat().st_size,
        'startup_index_seconds': index_seconds,
        'ownership_device_metadata_bytes': (ranker.owner_rows.numel() + ranker.owner_positions.numel()) * 8,
        'ownership_result_transfer_bytes': {
            'host_reduction': len(sources) * 4, 'device_reduction': TOP_K * 3 * 4,
        },
    }
    return ranker, db, {q: text(papers[q]) for q in queries}, labels, metadata


def request(ranker, connection, encoder, query, method):
    started = time.perf_counter()
    decoded = json.loads(query)
    vector = encoder.encode([decoded['query']], query=True)[0]
    encoded = time.perf_counter()
    scores = ranker.score(vector)
    rank = ranker.rank_host(scores) if method == 'host_reduction' else ranker.rank_device(scores)
    retrieved = time.perf_counter()
    ids = [ranker.index.source_ids[source] for _, source, _ in rank]
    placeholders = ','.join('?' for _ in ids)
    evidence = {source: (title, excerpt) for source, title, excerpt in connection.execute(
        f'SELECT id, title, excerpt FROM source WHERE id IN ({placeholders})', ids
    )}
    hits = []
    for (owner, _, score), source in zip(rank, ids, strict=True):
        expert = ranker.index.owner_ids[owner]
        title, excerpt = evidence[source]
        hits.append({'expert': expert, 'source': source, 'score': score, 'title': title, 'excerpt': excerpt})
    delivered = json.dumps(hits, ensure_ascii=False, separators=(',', ':')).encode()
    finished = time.perf_counter()
    return delivered, [row['expert'] for row in hits], {
        'encode': encoded - started, 'search': retrieved - encoded,
        'hydrate_and_serialize': finished - retrieved, 'total': finished - started,
    }


def benchmark(ranker, connection, encoder, queries, labels):
    cases = [[qid for qid in queries] for _ in range(REPEATS)]
    rng = random.Random(SEED)
    for trial in cases:
        rng.shuffle(trial)
    payloads = {qid: json.dumps({'query': query}, ensure_ascii=False) for qid, query in queries.items()}
    for method in METHODS:
        request(ranker, connection, encoder, next(iter(payloads.values())), method)
    results = {qid: {'arms': {method: [] for method in METHODS}} for qid in queries}
    for r, trial in enumerate(cases):
        for i, qid in enumerate(trial):
            order = METHODS if (i + r) % 2 == 0 else METHODS[::-1]
            outputs = {
                method: request(ranker, connection, encoder, payloads[qid], method) for method in order
            }
            if outputs[METHODS[0]][0] != outputs[METHODS[1]][0]:
                raise ValueError(f'Different exact outputs for {qid}')
            row = results[qid]
            payload, ranking, _ = outputs[METHODS[0]]
            digest = hashlib.sha256(payload).hexdigest()
            if 'response_sha256' in row and row['response_sha256'] != digest:
                raise ValueError(f'Unstable exact output for {qid}')
            row['response_sha256'] = digest
            row['ranking'] = ranking
            row['quality'] = {
                task: {
                    'positive': sum(by_query[qid].get(expert) == 1 for expert in ranking),
                    'judged': sum(expert in by_query[qid] for expert in ranking),
                } for task, by_query in labels.items()
            }
            for method, (_, _, timing) in outputs.items():
                row['arms'][method].append(timing)
    return results


def summarize(rows):
    results = {}
    for method in METHODS:
        timings = [t for row in rows.values() for t in row['arms'][method]]
        results[method] = {
            'mean_ms': {part: 1000 * mean(t[part] for t in timings) for part in timings[0]},
            'p50_total_ms': float(np.percentile([1000 * t['total'] for t in timings], 50)),
            'p95_total_ms': float(np.percentile([1000 * t['total'] for t in timings], 95)),
        }
    delta = [
        1000 * (mean(t['total'] for t in r['arms']['device_reduction'])
                - mean(t['total'] for t in r['arms']['host_reduction']))
        for r in rows.values()
    ]
    return {
        'runs_per_method': len(rows) * REPEATS, 'arms': results,
        'device_minus_host_total_ms': paired_interval(delta),
        'shared_quality': {
            task: {
                'known_positive_at_5': mean(row['quality'][task]['positive'] for row in rows.values()) / TOP_K,
                'judged_coverage_at_5': mean(row['quality'][task]['judged'] for row in rows.values()) / TOP_K,
            } for task in next(iter(rows.values()))['quality']
        },
    }


def main(output, datasets):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model_manifest = EMBEDDING / 'checkpoint-provenance.json'
    for file, expected in json.loads(model_manifest.read_text())['files'].items():
        if sha(EMBEDDING / file) != expected:
            raise ValueError(f'Changed embedding checkpoint: {file}')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    encoder = QwenTorchEmbedding(max_length=512, batch_size=1)
    torch.cuda.synchronize()
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'model_init_seconds': time.perf_counter() - started,
        'model_manifest_sha256': sha(model_manifest),
        'encoder_sha256': hashlib.sha256(inspect.getsource(QwenTorchEmbedding).encode()).hexdigest(),
        'script_sha256': sha(Path(__file__)), 'datasets': {},
        'scope': (
            'Frozen model and corpus. Warm sequential requests under both arms include reading a '
            'JSON query, live encoding, exact source ranking with maximum owner aggregation, reading '
            'owned excerpts through local SQLite, and response serialization. Both arms share GPU '
            'source scoring; only ownership reduction and transfer change. No service/network load '
            'or semantic guarantee for citations. CoF labels are sparse in NIPS and represent '
            'reviewer suitability, not independently adjudicated evidence. Transfer counts are '
            'for reduction outputs only, excluding query encoding and device-to-device operations.'
        ),
    }
    for dataset in datasets:
        ranker, db, queries, labels, metadata = prepare(dataset, output, model_manifest)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(ranker, connection, encoder, queries, labels)
        finally:
            db.unlink()
        raw = output / f'{dataset}-rows.json'
        raw.write_text(json.dumps(rows, indent=2) + '\n')
        report['datasets'][dataset] = {**metadata, **summarize(rows), 'rows_sha256': sha(raw)}
        del ranker
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=('SIGIR', 'NIPS'), default=('SIGIR', 'NIPS'))
    args = parser.parse_args()
    main(args.output, args.datasets)
