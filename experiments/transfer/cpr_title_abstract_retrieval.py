"""Benchmark open-release CPR retrieval with a title-first abstract cascade."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.evidencebench_pilot import bm25
from experiments.hotpotqa.semantic_complement import verify_model
from experiments.decision.hotpot_evidence_action import sha

REVISION = '543c4f331e740a58b1988b026d63816d7a3bd24f'
INPUTS = {
    'eval.jsonl': '7e9a47085643e19e17962dd4808792ddd582d8ee05b5312f52a8315314c0d763',
    'papers.jsonl': 'ff2a0a10f742db09f90e84ae04a8e362153ce354b2ef74daa901fade0e2e295f',
}
INSTRUCTIONS = {
    'relevance': 'Retrieve a research paper whose title and abstract are relevant to the problem.',
    'transfer': ('Retrieve a research paper on a different topic whose method or formulation can '
                 'be transferred to solve the research problem through shared functional roles and relations.'),
}
METHODS = ('bm25_title', 'dense_title', 'transfer_title', 'transfer_mixed', 'cascade20')
STRATA = ('q_side', 'pos_side', 'other')
CAP = 20


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def inventory(folder):
    for name, expected in INPUTS.items():
        if sha(folder / name) != expected:
            raise ValueError(f'Changed public release: {name}')
    papers = {row['corpusid']: row for row in read(folder / 'papers.jsonl')}
    queries = read(folder / 'eval.jsonl')
    if len(queries) != 1300 or len({r['query_id'] for r in queries}) != 1300:
        raise ValueError('Unexpected query inventory')
    if len(papers) != 188475 or Counter(Counter(r['field'] for r in queries).values()) != {50: 26}:
        raise ValueError('Unexpected paper or field inventory')
    ids = set()
    for row in queries:
        pos = row['positive']['corpusid']
        negatives = row['negatives']
        if Counter(x['stratum'] for x in negatives) != dict.fromkeys(STRATA, 50):
            raise ValueError('Unexpected negative strata')
        negatives = {item['corpusid'] for item in negatives}
        if len(negatives) != 150 or pos in negatives or row['q_corpusid'] in negatives | {pos}:
            raise ValueError('Candidate duplicates or base paper leak')
        ids.update(negatives | {pos})
    if len(ids) != 187411:
        raise ValueError('Changed candidate inventory')
    return queries, {key: papers[key] for key in sorted(ids)}


def build(text, encoder, name):
    started = time.perf_counter()
    order = sorted(range(len(text)), key=lambda i: (len(text[i]), i))
    result = np.empty((len(text), 1024), dtype=np.float32)
    width = 4 * encoder.batch_size
    for start in range(0, len(order), width):
        members = order[start:start + width]
        result[members] = encoder.encode([text[i] for i in members])
        if start and start % (width * 100) == 0:
            print(name, start, '/', len(text), flush=True)
    return result, time.perf_counter() - started


def dense_rank(matrix, query, selected):
    scores = matrix[selected] @ query
    return np.argsort(-scores, kind='stable')


def score(row, papers, positions, title_matrix, mixed_matrix, encoder):
    pos = row['positive']['corpusid']
    pool = {item['corpusid']: item['stratum'] for item in row['negatives']}
    keys = sorted([pos, *pool])
    positions_in_index = np.asarray([positions[key] for key in keys])
    gold = keys.index(pos)
    query = row['query']
    titles = [papers[key]['title'] for key in keys]
    started = time.perf_counter()
    lexical = bm25(query, titles)
    lexical_ms = (time.perf_counter() - started) * 1000

    vectors, encode_ms = {}, {}
    for name, instruction in INSTRUCTIONS.items():
        started = time.perf_counter()
        vectors[name] = encoder.encode([f'Instruct: {instruction}\nQuery: {query}'])[0]
        encode_ms[name] = (time.perf_counter() - started) * 1000

    ranks, timing = {'bm25_title': lexical}, {'bm25_title': lexical_ms}
    for name, matrix, view in (
        ('dense_title', title_matrix, 'relevance'), ('transfer_title', title_matrix, 'transfer'),
        ('transfer_mixed', mixed_matrix, 'transfer'),
    ):
        started = time.perf_counter()
        ranks[name] = dense_rank(matrix, vectors[view], positions_in_index)
        timing[name] = encode_ms[view] + (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    shortlist = ranks['transfer_title'][:CAP]
    new_order = dense_rank(mixed_matrix, vectors['transfer'], positions_in_index[shortlist])
    ranks['cascade20'] = np.concatenate((shortlist[new_order], ranks['transfer_title'][CAP:]))
    timing['cascade20'] = timing['transfer_title'] + (time.perf_counter() - started) * 1000
    outcomes = {}
    for name, ranking in ranks.items():
        ordered = list(map(int, ranking))
        rank = ordered.index(gold) + 1
        counts = Counter(pool[keys[i]] for i in ordered[:rank - 1])
        outcomes[name] = {'rank': rank, 'ms': timing[name],
                          'negative_types_ahead': {k: counts[k] for k in STRATA}}
    return {'id': row['query_id'], 'field': row['field'], 'positive_abstract': bool(papers[pos]['abstract']),
            'arms': outcomes}


def summary(rows):
    means = {}
    for name in METHODS:
        outcomes = [row['arms'][name] for row in rows]
        ranks = np.asarray([item['rank'] for item in outcomes])
        means[name] = {f'recall@{k}': float(np.mean(ranks <= k)) for k in (1, 5, 10, CAP)}
        means[name]['mrr@10'] = float(np.mean(np.where(ranks <= 10, 1 / ranks, 0)))
        means[name]['p95_ms'] = float(np.quantile([o['ms'] for o in outcomes], .95))
        means[name]['mean_negatives_ahead'] = {k: float(np.mean([o['negative_types_ahead'][k] for o in outcomes]))
                                                for k in STRATA}
    deltas = {method: paired_interval([int(r['arms'][method]['rank'] <= 10)
                                       - int(r['arms']['dense_title']['rank'] <= 10) for r in rows], repeats=4000)
              for method in ('transfer_title', 'transfer_mixed', 'cascade20')}
    return {'queries': len(rows), 'arms': means, 'paired_recall10_vs_dense_title': deltas}


def run(folder, output):
    import torch

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU allocation')
    queries, papers = inventory(folder)
    model_id = verify_model()
    encoder = QwenTorchEmbedding(max_length=512, batch_size=64)
    encoder.model.config.use_cache = False
    encoder.encode(['Warm up.'])
    paper_list = list(papers.values())
    positions = {record['corpusid']: i for i, record in enumerate(paper_list)}
    with_abstract = [i for i, p in enumerate(paper_list) if p['abstract']]
    if any(not paper_list[i]['abstract_redistributable'] for i in with_abstract):
        raise ValueError('Unexpected abstract license flag')
    title, title_seconds = build([p['title'] for p in paper_list], encoder, 'titles')
    full, abstract_seconds = build([
        paper_list[i]['title'] + '\n' + paper_list[i]['abstract'][:2000] for i in with_abstract
    ], encoder, 'abstracts')
    mixed = title.copy()
    mixed[with_abstract] = full
    del full
    output.mkdir(parents=True, exist_ok=False)
    score(queries[0], papers, positions, title, mixed, encoder)
    rows = [score(row, papers, positions, title, mixed, encoder) for row in queries]
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'release': REVISION, 'input_hashes': INPUTS, 'encoder_sha256': model_id,
        'code_sha256': sha(Path(__file__)), 'rows_sha256': sha(path),
        'index': {'candidates': len(paper_list), 'with_abstract': len(with_abstract),
                  'title_seconds': title_seconds, 'abstract_seconds': abstract_seconds,
                  'title_bytes': title.nbytes, 'both_views_bytes': title.nbytes + mixed.nbytes},
        'all': summary(rows),
        'positive_abstract': {str(flag).lower(): summary([r for r in rows if r['positive_abstract'] == flag])
                              for flag in (True, False)},
        'scope': 'All official CPR-eval candidate pools and redistributable abstracts only. '
                 'Items missing abstracts use their titles; no imputation. Fixed pools preclude corpus-level '
                 'latency comparisons. Report per-query scorer times, excluding index build, networking, '
                 'pool creation and response serialization. Paper transfer is not expert suitability.',
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
