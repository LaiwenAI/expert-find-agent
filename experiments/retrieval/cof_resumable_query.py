"""Measure shallow exit and native continuation without rerunning query layers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from statistics import mean
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_backbone_finetune import ROOT
from experiments.retrieval.cof_query_encoding_budget import TASK, QueryEncoderVariants, compare_labels
from expert_search.statistics import paired_interval

METHODS = ('full', 'shallow', 'adaptive', 'median_gate', 'always_continue')
REPEATS = 3
TOLERANCE = .02


class ResumableQuery:
    def __init__(self, variants, projection):
        self.variants = variants
        self.encoder = variants.encoder
        variants.configure('no_cache')
        self.model = self.encoder.model
        self.weights, self.bias = projection
        if len(self.model.layers) != 28 or self.model.has_sliding_layers:
            raise ValueError('Expected a 28-layer full-attention encoder')

    def inputs(self, query):
        prompted = 'Instruct: ' + TASK + '\nQuery:' + query
        batch = self.encoder.tokenizer(
            [prompted], max_length=512, truncation=True, padding=True, return_tensors='pt'
        ).to('cuda')
        hidden = self.model.embed_tokens(batch['input_ids'])
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        attention = create_causal_mask(
            config=self.model.config, inputs_embeds=hidden, attention_mask=batch['attention_mask'],
            position_ids=positions, past_key_values=None,
        )
        rotary = self.model.rotary_emb(hidden, positions)
        return hidden, (positions, attention, rotary)

    def advance(self, hidden, context, start, stop):
        positions, attention, rotary = context
        for layer in self.model.layers[start:stop]:
            hidden = layer(
                hidden, attention_mask=attention, position_embeddings=rotary,
                position_ids=positions, past_key_values=None, use_cache=False,
            )
        return hidden

    def finish(self, hidden, shallow=False):
        vector = F.normalize(self.model.norm(hidden)[:, -1, :].float(), dim=1).cpu().numpy()
        if shallow:
            vector = vector @ self.weights + self.bias
            vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        return vector[0]

    @torch.inference_mode()
    def check(self, texts):
        for query in texts:
            full_reference = self.encoder.encode([query], query=True)[0]
            self.variants.configure('layers_21')
            try:
                early_reference = self.encoder.encode([query], query=True)
            finally:
                self.variants.configure('no_cache')
            hidden, context = self.inputs(query)
            early = self.advance(hidden, context, 0, 21)
            reference = early_reference @ self.weights + self.bias
            reference /= np.linalg.norm(reference, axis=1, keepdims=True)
            if not np.array_equal(reference[0], self.finish(early, True)):
                raise ValueError('Resumable shallow encoder differs from baseline')
            if not np.array_equal(full_reference, self.finish(self.advance(early, context, 21, 28))):
                raise ValueError('Resumed full encoder differs from baseline')


def retrieve(index, vector, eligible=None):
    scores = index.score(vector).cpu().numpy()
    reference = index.index
    maxima = np.maximum.reduceat(scores[reference.owner_rows], reference.owner_starts)
    ordered = np.argsort(-maxima, kind='stable')
    if eligible is not None:
        # Membership affects returned ranking only. The exit decision uses the global margin.
        candidates = [i for i in ordered if reference.owner_ids[i] in eligible][:TOP_K]
    else:
        candidates = ordered[:TOP_K]
    rank = []
    for i in candidates:
        rows = reference.owners[reference.owner_ids[i]]
        witness = rows[int(np.argmax(scores[rows]))]
        rank.append((int(i), int(witness), float(maxima[i])))
    return rank, float(maxima[ordered[TOP_K - 1]] - maxima[ordered[TOP_K]])


@torch.inference_mode()
def request(query, encoder, index, connection, arm, threshold=0.0, eligible=None):
    started = time.perf_counter()
    hidden, context = encoder.inputs(json.loads(query)['query'])
    early_exit = False
    margin = None
    if arm == 'full':
        rank, _ = retrieve(index, encoder.finish(encoder.advance(hidden, context, 0, 28)), eligible)
    else:
        early = encoder.advance(hidden, context, 0, 21)
        rank, margin = retrieve(index, encoder.finish(early, shallow=True), eligible)
        early_exit = arm == 'shallow' or margin >= threshold
        if not early_exit:
            rank, _ = retrieve(index, encoder.finish(encoder.advance(early, context, 21, 28)), eligible)
    data, experts = serialize_hits(index, connection, rank)
    elapsed = time.perf_counter() - started
    return hashlib.sha256(data).hexdigest(), experts, margin, early_exit, elapsed


def payload(text):
    return json.dumps({'query': text}, ensure_ascii=False)


def calibrate(ranker, db, queries, labels, encoder):
    records = []
    with sqlite3.connect(db) as connection:
        for qid, query in queries.items():
            full = request(payload(query), encoder, ranker, connection, 'full')
            early = request(payload(query), encoder, ranker, connection, 'shallow')
            delta = [compare_labels(full[1], early[1], labels[grade][qid])['known_difference']
                     for grade in ('soft', 'hard')]
            records.append((early[2], delta))
    thresholds = sorted({margin for margin, _ in records}) + [float('inf')]
    for threshold in thresholds:
        accepted = [diff for margin, diff in records if margin >= threshold]
        differences = [sum(pair[i] for pair in accepted) / len(records) for i in range(2)]
        if min(differences) >= -TOLERANCE:
            return threshold, {
                'queries': len(records), 'accepted': len(accepted),
                'observed_soft_hard_delta': differences, 'allowed_per_metric_loss': TOLERANCE,
                'threshold': None if not np.isfinite(threshold) else threshold,
                'median_margin': float(np.median([margin for margin, _ in records])),
            }
    raise AssertionError('Always reverting to full inference must be feasible')


def run_test(ranker, db, queries, labels, encoder, thresholds, eligible=None):
    results = {q: {arm: {'times': []} for arm in METHODS} for q in queries}
    with sqlite3.connect(db) as connection:
        first = next(iter(queries))
        for arm in METHODS:
            request(payload(queries[first]), encoder, ranker, connection, arm, thresholds.get(arm, 0.),
                    None if eligible is None else eligible[first])
        for repeat in range(REPEATS):
            for pos, (qid, query) in enumerate(queries.items()):
                pool = None if eligible is None else eligible[qid]
                offset = (repeat + pos) % len(METHODS)
                for arm in METHODS[offset:] + METHODS[:offset]:
                    digest, experts, margin, exited, seconds = request(
                        payload(query), encoder, ranker, connection, arm, thresholds.get(arm, 0.), pool
                    )
                    observation = results[qid][arm]
                    if observation.get('digest', digest) != digest:
                        raise ValueError(f'Unstable result: {qid}/{arm}')
                    observation.update(digest=digest, experts=experts, exited=exited, margin=margin)
                    observation['times'].append(seconds)
                row = results[qid]
                for arm in thresholds:
                    expected = 'shallow' if row[arm]['exited'] else 'full'
                    if row[arm]['digest'] != row[expected]['digest']:
                        raise ValueError(f'Resume output differs on {qid}/{arm}')
    summary = {}
    for arm in METHODS:
        values = [r[arm] for r in results.values()]
        milliseconds = [1000 * t for row in values for t in row['times']]
        summary[arm] = {
            'early_exits': sum(row['exited'] for row in values),
            'mean_request_ms': mean(milliseconds),
            'p95_request_ms': float(np.quantile(milliseconds, .95)),
            'quality': {},
        }
        for grade, judgments in labels.items():
            pairs = [compare_labels(row['full']['experts'], row[arm]['experts'], judgments[q])
                     for q, row in results.items()]
            summary[arm]['quality'][grade] = {
                **{key: mean(row[key] for row in pairs) for key in pairs[0]},
                'known_paired_interval': paired_interval([m['known_difference'] for m in pairs]),
            }
    return results, summary


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    baseline = json.loads((ROOT / 'summary.json').read_text())
    projection = ROOT / 'projection.npz'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed checkpoint file: {name}')
    if sha(checkpoint) != baseline['checkpoint_manifest_sha256'] or sha(projection) != baseline['projector_sha256']:
        raise ValueError('Changed locked model or projection')
    with np.load(projection, allow_pickle=False) as saved:
        encoder = ResumableQuery(QueryEncoderVariants(), (saved['weights'], saved['bias']))

    output.mkdir(parents=True, exist_ok=False)
    ranker, db, queries, labels, _ = prepare('SIGIR', output, checkpoint)
    encoder.check(list(queries.values())[:8])
    threshold, calibration = calibrate(ranker, db, queries, labels, encoder)
    db.unlink()
    del ranker
    drop_queries, drop_sources, exclusion = exclusions()
    ranker, db, queries, labels, metadata = prepare(
        DATASET, output, checkpoint, exclude_queries=drop_queries, exclude_sources=drop_sources
    )
    encoder.check(list(queries.values())[:8])
    eligible = {q: labels['soft'][q].keys() & ranker.index.owners.keys() for q in queries}
    if any(not pool for pool in eligible.values()):
        raise ValueError('Empty judged candidate pool')
    if any(labels['soft'][q].keys() != labels['hard'][q].keys() for q in queries):
        raise ValueError('Prediction grades cover different candidate inventories')
    thresholds = {
        'adaptive': threshold,
        'median_gate': calibration['median_margin'],
        'always_continue': float('inf'),
    }
    results = {}
    for setting, pool in (('global', None), ('judged_pool', eligible)):
        rows, metrics = run_test(ranker, db, queries, labels, encoder, thresholds, pool)
        path = output / f'{setting}-rows.json'
        path.write_text(json.dumps(rows, indent=2) + '\n')
        results[setting] = {'metrics': metrics, 'rows_sha256': sha(path)}
    db.unlink()
    summary = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(checkpoint),
        'projection_sha256': sha(projection), 'calibration': calibration,
        'exclusions': exclusion, 'index_metadata': metadata, 'results': results,
        'scope': (
            'The global margin after 21 layers is the difference between fifth and sixth expert '
            'scores. In-domain SIGIR labels calibrate an exploratory point loss tolerance; '
            'disjoint filtered SciRepEval measures transfer. Label-blind median score margin '
            'and forced continuation are added as operating-point controls. Fallback runs only the remaining '
            'seven layers plus a second retrieval, and retains exact full outputs. Timings warm '
            'serial local request processing with SQLite citations; network and verification excluded.'
        ),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
