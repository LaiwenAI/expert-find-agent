"""Fine-tune the last two query blocks against a fixed document index."""
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
from safetensors.torch import save_file

from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, TASK
from experiments.retrieval.cof_rank_distillation import ANCHORS, RIDGE_ROOT, TEMPERATURE, embed
from experiments.retrieval.cof_shallow_alignment import (
    AlignedEncoder, CALIBRATION_ROWS, FIT_ROWS, SEED, benchmark, public_training_texts, summarize,
)
from expert_search.statistics import paired_interval

ROOT = Path('artifacts/cof-rank-distillation-v1')
ARMS = {
    'full': ('no_cache', False),
    'projection_only': ('layers_21', True),
    'backbone_finetune': ('layers_21', True),
}
EPOCHS = 3
BATCH = 16
LEARNING_RATE = 1e-5


def supervision(teacher, documents, student, projection):
    weight, bias = (torch.as_tensor(v, device='cuda') for v in projection)
    teacher_scores = teacher @ documents.T
    student_scores = F.normalize(student @ weight + bias, dim=-1) @ documents.T
    teacher_best = teacher_scores[:FIT_ROWS].topk(32).indices.cpu().numpy()
    student_best = student_scores[:FIT_ROWS].topk(32).indices.cpu().numpy()
    random_docs = np.random.default_rng(SEED).integers(len(documents), size=(FIT_ROWS, 32))
    selected = np.sort(np.concatenate((teacher_best, student_best, random_docs), axis=1), axis=1)
    valid = np.ones_like(selected, dtype=bool)
    valid[:, 1:] = selected[:, 1:] != selected[:, :-1]
    indices = torch.as_tensor(selected, device='cuda')
    mask = torch.as_tensor(valid, device='cuda')
    targets = (teacher_scores[:FIT_ROWS].gather(1, indices) / TEMPERATURE).masked_fill(
        ~mask, float('-inf')
    ).softmax(dim=1)
    return indices, mask, targets, teacher_scores[FIT_ROWS:].topk(5).indices


def train(variants, queries, documents, projection, data):
    variants.configure('layers_21')
    encoder = variants.encoder
    model = encoder.model
    model.requires_grad_(False)
    trained = {f'layers.{layer}.{name}': parameter
               for layer in (19, 20) for name, parameter in model.layers[layer].named_parameters()}
    init = {name: value.detach().cpu().clone() for name, value in trained.items()}
    for value in trained.values():
        value.data = value.data.float()
        value.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(trained.values()), lr=LEARNING_RATE)
    weight, bias = (torch.as_tensor(v, device='cuda') for v in projection)
    indices, mask, targets, teacher_top5 = data
    prompts = ['Instruct: ' + TASK + '\nQuery:' + q for q in queries]
    # Cache exact token IDs during training only. The online benchmark tokenizes each request.
    tokenized = encoder.tokenizer(prompts, max_length=512, truncation=True, padding=False)['input_ids']

    def live(batch):
        encoded = encoder.tokenizer.pad(
            [{'input_ids': tokenized[int(i)]} for i in batch], return_tensors='pt'
        ).to('cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            hidden = model(**encoded, use_cache=False).last_hidden_state[:, -1, :]
        return F.normalize(F.normalize(hidden.float(), dim=-1) @ weight + bias, dim=-1)

    def quality():
        with torch.no_grad():
            rank = torch.cat([
                (live(i) @ documents.T).topk(5).indices
                for i in np.array_split(np.arange(FIT_ROWS, len(queries)), CALIBRATION_ROWS // BATCH)
            ])
            return float((rank[:, :, None] == teacher_top5[:, None, :]).any(dim=2).float().mean())

    initial_quality = quality()
    rng = np.random.default_rng(SEED)
    history = []
    best = None
    best_quality = -1.
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for offset in np.array_split(rng.permutation(FIT_ROWS), FIT_ROWS // BATCH):
            batch = torch.as_tensor(offset, device='cuda')
            query = live(offset)
            scores = torch.einsum('bd,bkd->bk', query, documents[indices[batch]]) / TEMPERATURE
            log_probs = scores.masked_fill(~mask[batch], float('-inf')).log_softmax(dim=-1)
            loss = -(targets[batch] * log_probs.masked_fill(~mask[batch], 0)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trained.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        score = quality()
        history.append({'epoch': epoch, 'overlap': score, 'mean_loss': float(torch.stack(losses).mean())})
        print(history[-1], flush=True)
        if score > best_quality:
            best_quality = score
            best = {name: value.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
                    for name, value in trained.items()}
    with torch.no_grad():
        for name, value in trained.items():
            value.data = best[name].to('cuda')
            value.requires_grad_(False)
    deploy_quality = quality()
    changed = sum(int((best[k] != init[k]).sum()) for k in best)
    if not changed:
        raise RuntimeError('Fine-tuned weights match initialization')
    return best, {
        'trainable_parameters': sum(value.numel() for value in trained.values()),
        'changed_bf16_parameters': changed,
        'initial_calibration_overlap': initial_quality,
        'deployed_calibration_overlap': deploy_quality,
        'selected_epoch': max(history, key=lambda row: row['overlap'])['epoch'],
        'training_history': history,
    }


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for file, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / file) != expected:
            raise ValueError(f'Changed checkpoint file: {file}')
    original = json.loads((ROOT / 'summary.json').read_text())
    ridge = json.loads((RIDGE_ROOT / 'summary.json').read_text())
    projection_path = ROOT / 'projection.npz'
    if sha(checkpoint) != original['checkpoint_manifest_sha256'] or sha(projection_path) != original['projector_sha256']:
        raise ValueError('Changed projection or checkpoint')
    with np.load(projection_path, allow_pickle=False) as saved:
        projection = saved['weights'], saved['bias']
    count = FIT_ROWS + CALIBRATION_ROWS
    texts = public_training_texts(ANCHORS)
    queries, anchors = texts[:count], texts[count:]
    for d, expected in ((queries, ridge['training']['training_queries_sha256']), (anchors, original['anchors_sha256'])):
        if hashlib.sha256(json.dumps(d).encode()).hexdigest() != expected:
            raise ValueError('Changed corpus selection')

    output.mkdir(parents=True, exist_ok=False)
    variants = QueryEncoderVariants()
    started = time.perf_counter()
    teacher, documents, student = embed(variants, queries, anchors)
    data = supervision(teacher, documents, student, projection)
    torch.cuda.synchronize()
    embedded = time.perf_counter()
    del teacher, student
    weights, metrics = train(variants, queries, documents, projection, data)
    trained = time.perf_counter()
    model_file = output / 'query_blocks_20_21.safetensors'
    save_file(weights, str(model_file))
    frozen = QueryEncoderVariants()
    projections = {
        'projection_only': AlignedEncoder(frozen.encoder, *projection),
        'backbone_finetune': AlignedEncoder(variants.encoder, *projection),
    }
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'checkpoint_manifest_sha256': sha(checkpoint), 'script_sha256': sha(Path(__file__)),
        'weights_sha256': sha(model_file), 'training': metrics,
        'embedding_and_supervision_seconds': embedded - started, 'finetune_seconds': trained - embedded,
        'fit_queries': FIT_ROWS, 'calibration_queries': CALIBRATION_ROWS, 'anchor_papers': ANCHORS, 'datasets': {},
        'scope': (
            'Last two blocks of the 21-layer query encoding path fine-tuned on KDD teacher scores. '
            'All other model weights, the projection and document indices are frozen. Select epoch '
            'using KDD document overlap, then evaluate a BF16 checkpoint in the same request harness. '
            'No reviewer labels used in training. Reused CoF cohorts; NIPS has sparse grades. '
            'Warm sequential request duration excludes network and semantic citation verification.'
        ),
    }
    for dataset in ('SIGIR', 'NIPS'):
        ranker, db, test, labels, info = prepare(dataset, output, checkpoint)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(frozen, projections, ranker, connection, test, labels, arms=ARMS)
        finally:
            db.unlink()
        saved = ROOT / f'{dataset}-rows.json'
        if sha(saved) != original['datasets'][dataset]['rows_sha256']:
            raise ValueError('Changed earlier rows')
        earlier = json.loads(saved.read_text())
        if earlier.keys() != rows.keys() or any(
            row['projection_only']['rank'] != earlier[q]['distilled_21']['rank'] for q, row in rows.items()
        ):
            raise ValueError('Projection baseline ranking changed')
        comparison = {}
        for task in ('soft', 'hard'):
            comparison[task] = paired_interval([
                row['backbone_finetune']['quality'][task]['known_positive_at_5']
                - row['projection_only']['quality'][task]['known_positive_at_5']
                for row in rows.values()
            ])
        path = output / f'{dataset}-rows.json'
        path.write_text(json.dumps(rows, indent=2) + '\n')
        report['datasets'][dataset] = {
            **info, 'rows_sha256': sha(path), 'results': summarize(rows, arms=ARMS),
            'finetune_minus_projection': comparison,
        }
        del ranker
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
