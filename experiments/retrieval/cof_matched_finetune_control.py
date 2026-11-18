"""Compare trained query blocks to an equal-step update of the query projection."""
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
from safetensors.torch import load_file

from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_backbone_finetune import (
    ARMS as BASE_ARMS, BATCH, EPOCHS, LEARNING_RATE, ROOT, supervision,
)
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels
from experiments.retrieval.cof_rank_distillation import ANCHORS, RIDGE_ROOT, TEMPERATURE, embed
from experiments.retrieval.cof_shallow_alignment import (
    AlignedEncoder, CALIBRATION_ROWS, FIT_ROWS, SEED, benchmark, public_training_texts, summarize,
)
from expert_search.statistics import paired_interval

NATIVE = Path('artifacts/cof-query-backbone-finetune-v1')
ARMS = {**BASE_ARMS, 'matched_projection': ('layers_21', True)}


def optimize(student, documents, initial, supervision_data):
    indices, mask, teacher_probs, teacher_top5 = supervision_data
    weight, bias = [torch.nn.Parameter(torch.as_tensor(x, device='cuda').clone()) for x in initial]
    parameters = [weight, bias]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    steps = []
    best = None
    best_score = -1.
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for offset in np.array_split(rng.permutation(FIT_ROWS), FIT_ROWS // BATCH):
            batch = torch.as_tensor(offset, device='cuda')
            q = F.normalize(student[batch] @ weight + bias, dim=-1)
            scores = torch.einsum('bd,bkd->bk', q, documents[indices[batch]]) / TEMPERATURE
            probs = scores.masked_fill(~mask[batch], float('-inf')).log_softmax(dim=-1)
            loss = -(teacher_probs[batch] * probs.masked_fill(~mask[batch], 0)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        with torch.no_grad():
            rank = (F.normalize(student[FIT_ROWS:] @ weight + bias, dim=-1) @ documents.T).topk(5).indices
            quality = float((rank[:, :, None] == teacher_top5[:, None, :]).any(dim=2).float().mean())
            steps.append({'epoch': epoch, 'mean_loss': float(torch.stack(losses).mean()), 'overlap': quality})
            if quality > best_score:
                best_score = quality
                best = tuple(w.detach().cpu().numpy().copy() for w in parameters)
    return best, steps


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = EMBEDDING / 'checkpoint-provenance.json'
    for file, expected in json.loads(manifest.read_text())['files'].items():
        if sha(EMBEDDING / file) != expected:
            raise ValueError(f'Changed model file: {file}')
    native = json.loads((NATIVE / 'summary.json').read_text())
    original = json.loads((ROOT / 'summary.json').read_text())
    ridge = json.loads((RIDGE_ROOT / 'summary.json').read_text())
    weights = NATIVE / 'query_blocks_20_21.safetensors'
    original_projection = ROOT / 'projection.npz'
    if (sha(weights) != native['weights_sha256'] or sha(manifest) != native['checkpoint_manifest_sha256']
            or sha(original_projection) != original['projector_sha256']):
        raise ValueError('Changed reference weights')
    with np.load(original_projection, allow_pickle=False) as saved:
        initial = saved['weights'], saved['bias']
    count = FIT_ROWS + CALIBRATION_ROWS
    texts = public_training_texts(ANCHORS)
    queries, anchors = texts[:count], texts[count:]
    for values, expected in ((queries, ridge['training']['training_queries_sha256']), (anchors, original['anchors_sha256'])):
        if hashlib.sha256(json.dumps(values).encode()).hexdigest() != expected:
            raise ValueError('Changed documents or query selection')

    output.mkdir(parents=True, exist_ok=False)
    frozen = QueryEncoderVariants()
    started = time.perf_counter()
    teacher, docs, student = embed(frozen, queries, anchors)
    labels = supervision(teacher, docs, student, initial)
    torch.cuda.synchronize()
    embedded = time.perf_counter()
    projection, steps = optimize(student, docs, initial, labels)
    training_seconds = time.perf_counter() - embedded
    del teacher, docs, student, labels
    projection_file = output / 'matched_projection.npz'
    np.savez(projection_file, weights=projection[0], bias=projection[1])

    variant = QueryEncoderVariants()
    variant.configure('layers_21')
    updated = load_file(str(weights))
    with torch.no_grad():
        parameters = dict(variant.encoder.model.named_parameters())
        for name, value in updated.items():
            parameters[name].copy_(value)
    encoders = {
        'projection_only': AlignedEncoder(frozen.encoder, *initial),
        'matched_projection': AlignedEncoder(frozen.encoder, *projection),
        'backbone_finetune': AlignedEncoder(variant.encoder, *initial),
    }
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'backbone_checkpoint_sha256': sha(weights),
        'projection_sha256': sha(projection_file), 'training': steps,
        'selected_epoch': max(steps, key=lambda row: row['overlap'])['epoch'],
        'training_seconds': training_seconds, 'embedding_and_supervision_seconds': embedded - started,
        'schedules': {'epochs': EPOCHS, 'batch_size': BATCH, 'learning_rate': LEARNING_RATE},
        'datasets': {},
        'scope': (
            'Additional projection-only fitting uses the same target lists, learning rate, batch '
            'order, clipping and epoch budget as block fine-tuning. Only the trainable parameter '
            'set differs. Calibration independently selects each epoch on KDD. Timing is a single '
            'warm request, with frozen candidate indices and without network or citation inference.'
        ),
    }
    for corpus in ('SIGIR', 'NIPS'):
        ranker, db, test, grades, metadata = prepare(corpus, output, manifest)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(frozen, encoders, ranker, connection, test, grades, arms=ARMS)
        finally:
            db.unlink()
        old_path = NATIVE / f'{corpus}-rows.json'
        if sha(old_path) != native['datasets'][corpus]['rows_sha256']:
            raise ValueError('Changed native training output')
        earlier = json.loads(old_path.read_text())
        if earlier.keys() != rows.keys() or any(
            row['backbone_finetune']['response_sha256'] != earlier[q]['backbone_finetune']['response_sha256']
            for q, row in rows.items()
        ):
            raise ValueError('Saved BF16 backbone does not reproduce responses')
        comparisons = {}
        for task, qrels in grades.items():
            differences = [
                compare_labels(r['matched_projection']['rank'], r['backbone_finetune']['rank'], qrels[q])
                for q, r in rows.items()
            ]
            comparisons[task] = {
                'known_difference': paired_interval([d['known_difference'] for d in differences]),
                'lower': float(np.mean([d['difference_lower'] for d in differences])),
                'upper': float(np.mean([d['difference_upper'] for d in differences])),
            }
        raw = output / f'{corpus}-rows.json'
        raw.write_text(json.dumps(rows, indent=2) + '\n')
        result['datasets'][corpus] = {
            **metadata, 'rows_sha256': sha(raw), 'results': summarize(rows, arms=ARMS),
            'backbone_minus_matched_projection': comparisons,
        }
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
