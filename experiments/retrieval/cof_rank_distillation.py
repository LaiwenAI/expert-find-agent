"""Distill frozen-index neighbor scores into shallow expert-search query vectors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_shallow_alignment import (
    AlignedEncoder, CALIBRATION_ROWS, FIT_ROWS, SEED, benchmark,
    public_training_texts, summarize,
)

ARMS = {
    'full': ('no_cache', False),
    'ridge_21': ('layers_21', True),
    'distilled_21': ('layers_21', True),
}
ANCHORS = 4096
TEMPERATURE = .04
EPOCHS = 12
BATCH = 128
RIDGE_ROOT = Path('artifacts/cof-shallow-alignment-v1')


def embed(variants, training, anchors):
    encoder = variants.encoder
    encoder.batch_size = 16
    variants.configure('no_cache')
    teacher = encoder.encode(training, query=True)
    documents = encoder.encode(anchors, query=False)
    variants.configure('layers_21')
    student = encoder.encode(training, query=True)
    encoder.batch_size = 1
    return [torch.as_tensor(v, device='cuda') for v in (teacher, documents, student)]


def calibrate(student, documents, teacher_top5, weights, bias):
    projected = F.normalize(student @ weights + bias, dim=-1)
    top5 = (projected @ documents.T).topk(5).indices
    return float((top5[:, :, None] == teacher_top5[:, None, :]).any(dim=2).float().mean())


def distill(teacher, documents, student, initial):
    ridge_weights, ridge_bias = (torch.as_tensor(v, device='cuda') for v in initial)
    teacher_scores = teacher @ documents.T
    ridge_scores = F.normalize(student @ ridge_weights + ridge_bias, dim=-1) @ documents.T
    calibration_top5 = teacher_scores[FIT_ROWS:].topk(5).indices
    validation = student[FIT_ROWS:]
    ridge_overlap = calibrate(validation, documents, calibration_top5, ridge_weights, ridge_bias)

    positives = teacher_scores[:FIT_ROWS].topk(32).indices.cpu().numpy()
    hard = ridge_scores[:FIT_ROWS].topk(32).indices.cpu().numpy()
    rng = np.random.default_rng(SEED)
    random_rows = rng.integers(len(documents), size=(FIT_ROWS, 32))
    candidates = np.sort(np.concatenate((positives, hard, random_rows), axis=1), axis=1)
    distinct = np.ones_like(candidates, dtype=bool)
    distinct[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
    indices = torch.tensor(candidates, device='cuda')
    mask = torch.tensor(distinct, device='cuda')
    targets = (teacher_scores[:FIT_ROWS].gather(1, indices) / TEMPERATURE)
    targets = targets.masked_fill(~mask, float('-inf')).softmax(dim=-1)

    weights = torch.nn.Parameter(ridge_weights.clone())
    bias = torch.nn.Parameter(ridge_bias.clone())
    optimizer = torch.optim.AdamW([weights, bias], lr=2e-4)
    steps = []
    best = (-1.0, None, None)
    for epoch in range(1, EPOCHS + 1):
        for offset in np.array_split(rng.permutation(FIT_ROWS), FIT_ROWS // BATCH):
            batch = torch.tensor(offset, device='cuda')
            query = F.normalize(student[batch] @ weights + bias, dim=-1)
            scores = torch.einsum('bd,bkd->bk', query, documents[indices[batch]]) / TEMPERATURE
            log_prob = scores.masked_fill(~mask[batch], float('-inf')).log_softmax(dim=-1)
            loss = -(targets[batch] * log_prob.masked_fill(~mask[batch], 0)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            overlap = calibrate(validation, documents, calibration_top5, weights, bias)
            steps.append(overlap)
            if overlap > best[0]:
                best = overlap, weights.detach().clone(), bias.detach().clone()
    return (best[1].cpu().numpy(), best[2].cpu().numpy()), {
        'ridge_calibration_top5_overlap': ridge_overlap,
        'distilled_calibration_top5_overlap_by_epoch': steps,
        'selected_epoch': int(np.argmax(steps)) + 1,
        'selected_overlap': best[0],
        'mean_unique_training_candidates': float(distinct.sum(axis=1).mean()),
    }


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(manifest.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed checkpoint: {name}')
    training_count = FIT_ROWS + CALIBRATION_ROWS
    texts = public_training_texts(ANCHORS)
    training, anchors = texts[:training_count], texts[training_count:]
    prior = json.loads((RIDGE_ROOT / 'summary.json').read_text())
    if hashlib.sha256(json.dumps(training).encode()).hexdigest() != prior['training']['training_queries_sha256']:
        raise ValueError('Prior ridge training texts differ')
    path = RIDGE_ROOT / 'projections.npz'
    if sha(path) != prior['training']['projector_sha256'] or sha(manifest) != prior['checkpoint_manifest_sha256']:
        raise ValueError('Prior ridge artifact differs')
    with np.load(path, allow_pickle=False) as ridge:
        initial = ridge['layers_21_weights'], ridge['layers_21_bias']

    output.mkdir(parents=True, exist_ok=False)
    variants = QueryEncoderVariants()
    started = time.perf_counter()
    vectors = embed(variants, training, anchors)
    embedded = time.perf_counter()
    weights, training_report = distill(*vectors, initial)
    fitted = time.perf_counter()
    del vectors
    arrays = output / 'projection.npz'
    np.savez(arrays, weights=weights[0], bias=weights[1])
    projections = {
        'ridge_21': AlignedEncoder(variants.encoder, *initial),
        'distilled_21': AlignedEncoder(variants.encoder, *weights),
    }
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'seed': SEED,
        'ridge_sha256': sha(path), 'projector_sha256': sha(arrays),
        'fit_papers': FIT_ROWS, 'calibration_papers': CALIBRATION_ROWS, 'anchor_papers': ANCHORS,
        'anchors_sha256': hashlib.sha256(json.dumps(anchors).encode()).hexdigest(),
        'checkpoint_manifest_sha256': sha(manifest),
        'embedding_seconds': embedded - started, 'training_seconds': fitted - embedded,
        'training': training_report, 'arms': ARMS, 'datasets': {},
        'scope': (
            'KDD projected teacher query scores over a separate frozen KDD paper index. '
            'Train lists contain teacher top 32, ridge top 32 and 32 random documents, with '
            'duplicates masked. Epoch uses teacher top-five overlap on held-out KDD queries. '
            'No test labels or test indexed sources enter fitting. Corpus exclusion uses '
            'normalized titles and IDs. Warm request results use the same frozen published '
            'indices. NIPS quality remains partially identified. Reused evaluation cohorts.'
        ),
    }
    for dataset in ('SIGIR', 'NIPS'):
        ranker, db, queries, labels, metadata = prepare(dataset, output, manifest)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(variants, projections, ranker, connection, queries, labels, arms=ARMS)
        finally:
            db.unlink()
        file = output / f'{dataset}-rows.json'
        file.write_text(json.dumps(rows, indent=2) + '\n')
        result['datasets'][dataset] = {**metadata, 'rows_sha256': sha(file), 'results': summarize(rows, arms=ARMS)}
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
