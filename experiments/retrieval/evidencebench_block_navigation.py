"""Compare full-pool evidence ranking and query-independent source-block navigation."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.evidencebench_data import aspect_coverage, prepare
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.retrieval.cof_online_ownership import EMBEDDING, sha

ROOT = Path('artifacts/evidencebench-role-test-v1')
DATA = Path('artifacts/public/evidencebench/test.json')
TASK = 'Retrieve sentences containing concrete scientific evidence relevant to the hypothesis.'
BLOCK = 8
CAPS = (500, 1000)
ARMS = ('flat_dense', 'sparse_sentence', 'centroid_blocks', 'landmark_blocks', 'fusion_blocks')
REPEATS = 5
SEED = 20261121
WORDS = re.compile(r"\b[\w-]+\b")


def terms(text):
    return WORDS.findall(text.casefold())


class Sparse:
    """Build BM25 impact lists so requests only touch matching postings."""

    def __init__(self, blocks):
        counts = [Counter(terms(b)) for b in blocks]
        length = np.array([sum(row.values()) for row in counts])
        normalizer = 1.2 * (.25 + .75 * length / max(float(length.mean()), 1))
        postings = defaultdict(list)
        for pos, row in enumerate(counts):
            for word, freq in row.items():
                postings[word].append((pos, 2.2 * freq / (freq + normalizer[pos])))
        self.size = len(blocks)
        self.postings = {}
        for word, items in postings.items():
            ids, impacts = zip(*items, strict=True)
            idf = np.log1p((self.size - len(ids) + .5) / (len(ids) + .5))
            self.postings[word] = np.asarray(ids, dtype=np.int32), np.asarray(impacts, dtype=np.float32) * idf

    def score(self, query):
        result = np.zeros(self.size, dtype=np.float32)
        accesses = 0
        for word in query:
            if word in self.postings:
                ids, impacts = self.postings[word]
                result[ids] += impacts
                accesses += len(ids)
        return result, accesses

    @property
    def array_bytes(self):
        return sum(ids.nbytes + values.nbytes for ids, values in self.postings.values())


class PaperIndex:
    def __init__(self, sentences, vectors):
        self.texts = [s.strip() for s in sentences]
        self.cost = np.array([len(s) + 1 for s in self.texts], dtype=np.int32)
        self.vectors = vectors
        self.groups = [np.arange(start, min(start + BLOCK, len(self.texts)))
                       for start in range(0, len(self.texts), BLOCK)]
        self.width = max(1, (len(self.groups) + 3) // 4)
        self.leaf_cap = min(len(self.texts), self.width * BLOCK)
        started = perf_counter()
        self.centers = np.array([vectors[g].mean(axis=0) for g in self.groups], dtype=np.float32)
        self.centers /= np.maximum(np.linalg.norm(self.centers, axis=1, keepdims=True), 1e-12)
        centroid_s = perf_counter() - started
        started = perf_counter()
        landmarks = []
        for group, center in zip(self.groups, self.centers, strict=True):
            local = vectors[group]
            first = int(np.argmax(local @ center))
            other = int(np.argmin(local @ local[first]))
            landmarks.append(local[[first, other]])
        self.landmarks = np.asarray(landmarks, dtype=np.float32)
        landmark_s = centroid_s + perf_counter() - started
        started = perf_counter()
        self.sparse = Sparse(self.texts)
        sparse_s = perf_counter() - started
        started = perf_counter()
        self.block_sparse = Sparse([' '.join(self.texts[i] for i in group) for group in self.groups])
        self.build_seconds = {'centroids': centroid_s, 'landmarks_including_centroids': landmark_s,
                              'sparse_sentence': sparse_s, 'sparse_blocks': perf_counter() - started}

    def retrieve(self, method, embedding, query):
        candidates = None
        proxy = accesses = 0
        if method == 'sparse_sentence':
            scores, accesses = self.sparse.score(query)
            candidates = np.argsort(-scores, kind='stable')[:self.leaf_cap]
        elif method in ('centroid_blocks', 'landmark_blocks', 'fusion_blocks'):
            if method == 'landmark_blocks':
                scores = (self.landmarks.reshape(-1, embedding.size) @ embedding).reshape(-1, 2).max(axis=1)
                proxy = 2 * len(self.groups)
            else:
                scores = self.centers @ embedding
                proxy = len(self.groups)
            if method == 'fusion_blocks':
                sparse, accesses = self.block_sparse.score(query)
                offsets = np.arange(len(scores))
                dense_rank, sparse_rank = np.empty_like(offsets), np.empty_like(offsets)
                dense_rank[np.argsort(-scores, kind='stable')] = offsets
                sparse_rank[np.argsort(-sparse, kind='stable')] = offsets
                scores = 1 / (5 + dense_rank) + 1 / (5 + sparse_rank)
            blocks = np.argsort(-scores, kind='stable')[:self.width]
            candidates = np.concatenate([self.groups[i] for i in blocks])
        if candidates is None:
            ranking = np.argsort(-(self.vectors @ embedding), kind='stable')
            return ranking, len(ranking), proxy, accesses
        candidates.sort()
        ranking = candidates[np.argsort(-(self.vectors[candidates] @ embedding), kind='stable')]
        if len(candidates) > self.leaf_cap:
            raise ValueError('Leaf scoring exceeds the shared shortlist cap')
        return ranking, len(candidates), proxy, accesses

    def read(self, ranking, cap):
        selected, used = [], 0
        for i in ranking:
            if self.cost[i] > 1 and used + self.cost[i] <= cap + 1:
                selected.append(int(i))
                used += int(self.cost[i])
        characters = used - int(bool(selected))
        payload = '\n'.join(self.texts[i] for i in selected)
        if len(payload) != characters or characters > cap:
            raise ValueError('Invalid context budget')
        return selected, characters


def build(data, protocol):
    papers = {}
    query_to_index = {}
    all_hashes = hashlib.sha256()
    for key, row in sorted(data.items()):
        texts = row['paper_as_candidate_pool']
        source_sha = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        if protocol['source_sha256'][key] != source_sha:
            raise ValueError('Changed source sentences or order')
        paper = row['paper_id']
        if paper not in papers:
            path = ROOT / 'vectors' / f'{key}.npy'
            vectors = np.load(path, allow_pickle=False)
            if vectors.shape != (len(texts), 1024) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError('Invalid stored source vectors')
            all_hashes.update(f'{key}:{sha(path)}\n'.encode())
            papers[paper] = source_sha, PaperIndex(texts, vectors)
        if papers[paper][0] != source_sha:
            raise ValueError('Paper has different source snapshots')
        query_to_index[key] = papers[paper][1]
    return query_to_index, [index for _, index in papers.values()], all_hashes.hexdigest()


def evaluate(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, digest in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != digest:
            raise ValueError('Changed encoder')
    protocol_path = ROOT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    historical_path = ROOT / 'evaluation.json'
    verified = json.loads((ROOT / 'verification.json').read_text())['artifacts']
    if any(sha(path) != verified[str(path)] for path in (protocol_path, historical_path)):
        raise ValueError('Changed original experiment records')
    historical = {r['id']: r['baseline_selected'] for r in json.loads(historical_path.read_text())}
    data = prepare('test')
    if sha(DATA) != protocol['dataset_sha256'] or historical.keys() != data.keys():
        raise ValueError('Changed EvidenceBench snapshot')
    by_query, unique, vectors_hash = build(data, protocol)
    model = QwenTorchEmbedding(max_length=512, batch_size=1)
    sample = next(iter(data.values()))['hypothesis']
    model.encode(['Instruct: ' + TASK + '\nQuery: ' + sample])
    rng = random.Random(SEED)
    rows = []
    for key, row in sorted(data.items()):
        index = by_query[key]
        started = perf_counter()
        query = terms(row['hypothesis'])
        vector = model.encode(['Instruct: ' + TASK + '\nQuery: ' + row['hypothesis']])[0]
        shared_ms = (perf_counter() - started) * 1000
        trial = {'id': key, 'paper': row['paper_id'], 'shared_query_ms': shared_ms, 'arms': {}}
        for _ in range(REPEATS):
            order = list(ARMS)
            rng.shuffle(order)
            for arm in order:
                started = perf_counter()
                ranking, leaves, proxies, postings = index.retrieve(arm, vector, dict.fromkeys(query))
                primary = index.read(ranking, 1000)
                selection_ms = (perf_counter() - started) * 1000
                selections = [index.read(ranking, 500), primary]
                if arm not in trial['arms']:
                    trial['arms'][arm] = {
                        'leaves_scored': leaves, 'proxies_scored': proxies, 'postings_read': postings,
                        'reachable_aspects': aspect_coverage(row, ranking.tolist())['aspect_recall'],
                        'coverage': {str(cap): {**aspect_coverage(row, chosen), 'characters': used,
                                                'sentence_indices': chosen}
                                     for cap, (chosen, used) in zip(CAPS, selections, strict=True)}, 'ms': [],
                    }
                    if arm == 'flat_dense':
                        trial['historical_top10_matches'] = ranking[:10].tolist() == historical[key]
                entry = trial['arms'][arm]
                if any(chosen != entry['coverage'][str(cap)]['sentence_indices']
                       for cap, (chosen, _) in zip(CAPS, selections, strict=True)):
                    raise ValueError('Repeated selection changed')
                entry['ms'].append(selection_ms)
        rows.append(trial)
    metrics = {}
    for arm in ARMS:
        values = [r['arms'][arm] for r in rows]
        measurements = [(t, t + row['shared_query_ms']) for value, row in zip(values, rows, strict=True)
                        for t in value['ms']]
        selection_p95, total_p95 = np.quantile(measurements, .95, axis=0)
        metrics[arm] = {
            **{key: mean(v[key] for v in values)
               for key in ('leaves_scored', 'proxies_scored', 'postings_read', 'reachable_aspects')},
            'selection_p95_ms': float(selection_p95), 'query_plus_selection_p95_ms': float(total_p95),
            'caps': {str(cap): {key: mean(v['coverage'][str(cap)][key] for v in values)
                                for key in ('aspect_recall', 'characters')} for cap in CAPS},
        }
        for cap in map(str, CAPS):
            metrics[arm]['caps'][cap]['minus_flat'] = clustered_paired_interval(
                [v['coverage'][cap]['aspect_recall'] - row['arms']['flat_dense']['coverage'][cap]['aspect_recall']
                 for v, row in zip(values, rows, strict=True)], [r['paper'] for r in rows], repeats=4000)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(), 'queries': len(rows),
        'papers': len(unique), 'sentences': sum(len(p.texts) for p in unique),
        'historical_flat_top10_matches': sum(r['historical_top10_matches'] for r in rows),
        'primary_timed_cap': 1000,
        'data_sha256': sha(DATA), 'models_sha256': sha(checkpoint), 'code_sha256': sha(Path(__file__)),
        'rows_sha256': sha(path), 'stored_vector_inventory_sha256': vectors_hash,
        'build_seconds_from_cached_vectors': {
            key: sum(p.build_seconds[key] for p in unique) for key in unique[0].build_seconds},
        'array_bytes': {
            'centroids': sum(p.centers.nbytes for p in unique), 'landmarks': sum(p.landmarks.nbytes for p in unique),
            'sparse_sentence': sum(p.sparse.array_bytes for p in unique),
            'sparse_blocks': sum(p.block_sparse.array_bytes for p in unique),
        }, 'arms': metrics,
        'scope': 'Previously explored EvidenceBench test queries over known single-paper pools. One cached vector '
                 'snapshot per paper. Same trimmed characters and exact dense leaf scoring. No test relevance grades '
                 'in navigation. Extra index time excludes original source embedding, and the online P95 sums shared '
                 'query encoding and independently timed selection. No corpus retrieval, external service, or verification.',
    }
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args().output)
