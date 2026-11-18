"""Distill the frozen index's expert ranking after source ownership aggregation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels
from experiments.retrieval.cof_rank_distillation import ANCHORS, BATCH, EPOCHS, RIDGE_ROOT, TEMPERATURE, embed
from experiments.retrieval.cof_shallow_alignment import (
    AlignedEncoder, CALIBRATION_ROWS, FIT_ROWS, SEED, benchmark, public_training_texts, summarize,
)
from expert_search.statistics import paired_interval

ARMS = {
    'full': ('no_cache', False),
    'ridge_21': ('layers_21', True),
    'document_21': ('layers_21', True),
    'expert_21': ('layers_21', True),
}
DOCUMENT_ROOT = Path('artifacts/cof-rank-distillation-v1')


class Ownership:
    def __init__(self, anchor_ids, documents):
        by_paper = defaultdict(set)
        selected = set(anchor_ids)
        with ZipFile('artifacts/public/cof/data.zip') as release, release.open('data/KDD_reviewers_test.json') as stream:
            for line in stream:
                profile = json.loads(line)
                for paper in selected.intersection(profile['papers']):
                    by_paper[paper].add(profile['reviewer'])
        experts = sorted({expert for owners in by_paper.values() for expert in owners})
        positions = {expert: i for i, expert in enumerate(experts)}
        edges = [(i, positions[e]) for i, paper in enumerate(anchor_ids) for e in sorted(by_paper[paper])]
        if not edges or len(experts) < 32:
            raise ValueError('Insufficient KDD ownership records')
        rows, owners = zip(*edges, strict=True)
        self.documents = documents
        self.rows = torch.tensor(rows, device='cuda')
        self.owners = torch.tensor(owners, device='cuda')
        self.experts = len(experts)
        self.metadata = {
            'anchor_experts': len(experts),
            'ownership_edges': len(edges),
            'owned_anchor_papers': len(set(rows)),
            'ownership_sha256': hashlib.sha256(json.dumps([experts, edges]).encode()).hexdigest(),
        }

    def scores(self, queries):
        scores = (queries @ self.documents.T).index_select(1, self.rows)
        empty = scores.new_full((len(queries), self.experts), float('-inf'))
        return empty.scatter_reduce(1, self.owners.expand_as(scores), scores, reduce='amax')


def top_five(scores):
    return scores.argsort(dim=1, descending=True, stable=True)[:, :5]


def overlap(scores, target):
    choices = top_five(scores)
    return float((choices[:, :, None] == target[:, None, :]).any(dim=2).float().mean())


def fit(teacher, student, ownership, ridge, document):
    ridge_weight, ridge_bias = [torch.as_tensor(v, device='cuda') for v in ridge]
    doc_weight, doc_bias = [torch.as_tensor(v, device='cuda') for v in document]
    teacher_scores = ownership.scores(teacher)
    ridge_scores = ownership.scores(F.normalize(student @ ridge_weight + ridge_bias, dim=-1))
    target_top5 = top_five(teacher_scores[FIT_ROWS:])
    validation = student[FIT_ROWS:]
    ridge_overlap = overlap(ridge_scores[FIT_ROWS:], target_top5)
    doc_overlap = overlap(ownership.scores(F.normalize(validation @ doc_weight + doc_bias, dim=-1)),
                          target_top5)

    positives = teacher_scores[:FIT_ROWS].argsort(dim=1, descending=True, stable=True)[:, :32]
    hard = ridge_scores[:FIT_ROWS].argsort(dim=1, descending=True, stable=True)[:, :32]
    rng = np.random.default_rng(SEED)
    random_scores = rng.integers(ownership.experts, size=(FIT_ROWS, 32))
    candidates = np.sort(np.concatenate((positives.cpu().numpy(), hard.cpu().numpy(), random_scores), axis=1), axis=1)
    distinct = np.ones_like(candidates, dtype=bool)
    distinct[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
    indices = torch.as_tensor(candidates, device='cuda')
    mask = torch.as_tensor(distinct, device='cuda')
    targets = (teacher_scores[:FIT_ROWS].gather(1, indices) / TEMPERATURE)
    targets = targets.masked_fill(~mask, float('-inf')).softmax(dim=-1)

    weights = torch.nn.Parameter(ridge_weight.clone())
    bias = torch.nn.Parameter(ridge_bias.clone())
    optimizer = torch.optim.AdamW([weights, bias], lr=2e-4)
    scores_by_epoch = []
    best_score = -1.0
    best = None
    for _ in range(EPOCHS):
        for offset in np.array_split(rng.permutation(FIT_ROWS), FIT_ROWS // BATCH):
            batch = torch.as_tensor(offset, device='cuda')
            query = F.normalize(student[batch] @ weights + bias, dim=-1)
            local = ownership.scores(query).gather(1, indices[batch]) / TEMPERATURE
            log_probs = local.masked_fill(~mask[batch], float('-inf')).log_softmax(dim=-1)
            loss = -(targets[batch] * log_probs.masked_fill(~mask[batch], 0)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            current = overlap(ownership.scores(F.normalize(validation @ weights + bias, dim=-1)), target_top5)
            scores_by_epoch.append(current)
            if current > best_score:
                best_score = current
                best = weights.detach().cpu().numpy().copy(), bias.detach().cpu().numpy().copy()
    return best, {
        **ownership.metadata,
        'ridge_calibration_overlap': ridge_overlap, 'document_calibration_overlap': doc_overlap,
        'expert_calibration_overlap_by_epoch': scores_by_epoch,
        'selected_epoch': int(np.argmax(scores_by_epoch)) + 1,
        'mean_unique_training_candidates': float(distinct.sum(axis=1).mean()),
    }


def compare_expert_and_document(rows, labels):
    report = {}
    for task, grades in labels.items():
        deltas = [compare_labels(row['document_21']['rank'], row['expert_21']['rank'], grades[q])
                  for q, row in rows.items()]
        report[task] = {
            'known_paired_interval': paired_interval([m['known_difference'] for m in deltas]),
            'sparse_lower': float(np.mean([m['difference_lower'] for m in deltas])),
            'sparse_upper': float(np.mean([m['difference_upper'] for m in deltas])),
        }
    return report


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed model file: {name}')
    ridge_summary = json.loads((RIDGE_ROOT / 'summary.json').read_text())
    doc_summary = json.loads((DOCUMENT_ROOT / 'summary.json').read_text())
    count = FIT_ROWS + CALIBRATION_ROWS
    texts = public_training_texts(ANCHORS, include_ids=True)
    training = [value for value, _ in texts[:count]]
    anchors, anchor_ids = map(list, zip(*texts[count:], strict=True))
    if hashlib.sha256(json.dumps(training).encode()).hexdigest() != ridge_summary['training']['training_queries_sha256']:
        raise ValueError('Changed training texts')
    if hashlib.sha256(json.dumps(anchors).encode()).hexdigest() != doc_summary['anchors_sha256']:
        raise ValueError('Changed anchor texts')
    ridge_file = RIDGE_ROOT / 'projections.npz'
    doc_file = DOCUMENT_ROOT / 'projection.npz'
    if sha(checkpoint) != doc_summary['checkpoint_manifest_sha256'] or sha(ridge_file) != doc_summary['ridge_sha256'] or sha(doc_file) != doc_summary['projector_sha256']:
        raise ValueError('Projection or checkpoint provenance changed')
    with np.load(ridge_file, allow_pickle=False) as saved:
        ridge = saved['layers_21_weights'], saved['layers_21_bias']
    with np.load(doc_file, allow_pickle=False) as saved:
        document = saved['weights'], saved['bias']

    output.mkdir(parents=True, exist_ok=False)
    variants = QueryEncoderVariants()
    started = time.perf_counter()
    teacher, documents, student = embed(variants, training, anchors)
    embedded = time.perf_counter()
    ownership = Ownership(anchor_ids, documents)
    expert, metadata = fit(teacher, student, ownership, ridge, document)
    trained = time.perf_counter()
    del teacher, documents, student, ownership
    path = output / 'projection.npz'
    np.savez(path, weights=expert[0], bias=expert[1])
    projections = {
        'ridge_21': AlignedEncoder(variants.encoder, *ridge),
        'document_21': AlignedEncoder(variants.encoder, *document),
        'expert_21': AlignedEncoder(variants.encoder, *expert),
    }
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'projector_sha256': sha(path),
        'checkpoint_manifest_sha256': sha(checkpoint), 'document_projection_sha256': sha(doc_file),
        'training': metadata, 'embedding_seconds': embedded - started,
        'training_seconds': trained - embedded, 'datasets': {},
        'scope': (
            'Same KDD pseudoqueries, anchors, initialization and projection size as document '
            'distillation. Uses published KDD archive ownership rather than reviewer grades. '
            'Training and validation target the maximum source score per expert; calibration '
            'only uses held-out KDD pseudoqueries. Warm SIGIR and NIPS requests run on unchanged '
            'document indices and the original exact aggregation. Reused evaluation cohorts; '
            'NIPS grades are sparse. Timing excludes service/network and semantic verification.'
        ),
    }
    for dataset in ('SIGIR', 'NIPS'):
        ranker, db, queries, labels, info = prepare(dataset, output, checkpoint)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(variants, projections, ranker, connection, queries, labels, arms=ARMS)
        finally:
            db.unlink()
        baseline_file = DOCUMENT_ROOT / f'{dataset}-rows.json'
        if sha(baseline_file) != doc_summary['datasets'][dataset]['rows_sha256']:
            raise ValueError('Changed earlier document comparison')
        baseline = json.loads(baseline_file.read_text())
        if baseline.keys() != rows.keys() or any(
            row['document_21']['rank'] != baseline[q]['distilled_21']['rank'] for q, row in rows.items()
        ):
            raise ValueError('Failed to reproduce earlier document rankings')
        raw = output / f'{dataset}-rows.json'
        raw.write_text(json.dumps(rows, indent=2) + '\n')
        result['datasets'][dataset] = {
            **info, 'rows_sha256': sha(raw), 'results': summarize(rows, arms=ARMS),
            'expert_minus_document': compare_expert_and_document(rows, labels),
        }
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
