"""Train a one-pass compact stance decision with a calibrated abstention option."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_reranker import pack_tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.qasper.paired_reader import load_reader
from experiments.scifact import polarity_cascade as cascade
from experiments.scifact import polarity_gate as base

FOLDS = 4
RIDGE = 0.001
TASK = ("Determine whether the document supports or contradicts the scientific claim. "
        "Compare the entities, scope, measurements, and direction.")


class StanceFeatures(Reranker):
    @torch.inference_mode()
    def features(self, rows):
        sequences = []
        for row in rows:
            body = self.tokenizer.encode(
                f"<Instruct>: {TASK}\n<Query>: {row['criterion']}\n<Document>: {row['source']}",
                add_special_tokens=False,
            )
            packed, raw = pack_tokens(self.prefix, body, self.suffix, 1024)
            if raw != len(packed):
                raise ValueError("Unexpected clipped source input")
            sequences.append(dict(input_ids=packed))
        inputs = self.tokenizer.pad(sequences, padding=True, return_tensors="pt").to("cuda")
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        return torch.nn.functional.normalize(hidden.float()).cpu().numpy()


def partitions(rows):
    # Claims sharing an abstract are one connected component. Put
    # whole components in folds regardless of their stance labels.
    parent = {row["claim_id"]: row["claim_id"] for row in rows}
    def root(claim):
        while claim != parent[claim]:
            claim = parent[claim]
        return claim

    first = {}
    for row in rows:
        claim, doc = row["claim_id"], row["document"]
        parent[root(claim)] = root(first.setdefault(doc, claim))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(row["claim_id"])].append(i)
    folds = [[] for _ in range(FOLDS)]
    for group in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
        min(folds, key=len).extend(group)
    out = [np.asarray(sorted(fold)) for fold in folds]
    for fold in out:
        held = set(fold.tolist())
        for field in ("document", "claim_id"):
            first = {row[field] for i, row in enumerate(rows) if i in held}
            assert not first.intersection(row[field] for i, row in enumerate(rows) if i not in held)
    return out


def fit(features, gold):
    features = features.astype(np.float64)
    labels = 1 - 2 * gold
    x_center = features.mean(axis=0)
    y_center = labels.mean()
    x = features - x_center
    # Dual ridge uses the number of training rows times itself.
    matrix = x @ x.T + RIDGE * len(labels) * np.eye(len(labels))
    weights = x.T @ np.linalg.solve(matrix, labels - y_center)
    return weights, float(y_center - x_center @ weights)


def predict(features, model):
    scores = features @ model[0] + model[1]
    return np.column_stack((scores, -scores))


def timed(rows, operation):
    times, batches = [], []
    for start in range(0, len(rows), cascade.BATCH):
        begin = perf_counter()
        batches.append(operation(rows[start:start + cascade.BATCH]))
        times.append((perf_counter() - begin) * 1000)
    return np.concatenate(batches), times


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and a Slurm GPU")
    anchor = json.loads(args.reference.read_bytes())
    if digest(args.projection) != anchor["projection_sha256"]:
        raise ValueError("Changed evidence projection")
    rows = json.loads(args.projection.read_bytes())["rows"]
    dev, test = rows["calibration"], rows["test"]
    gold = {name: np.array([base.LABELS.index(r["label"]) for r in panel]) for name, panel in rows.items()}
    compact = StanceFeatures(args.reranker)
    tokenizer, large, provenance = load_reader(args.chat)
    if compact.provenance["revision"] != anchor["reranker_revision"] or provenance["revision"] != anchor["chat_revision"]:
        raise ValueError("Changed model checkpoints")

    x, _ = timed(dev, compact.features)
    y = gold["calibration"]
    out_of_fold = np.empty((len(y), 2))
    folds = partitions(dev)
    for held in folds:
        source = np.ones(len(y), dtype=bool)
        source[held] = False
        if not len(held) or len(set(y[source])) != 2:
            raise ValueError("Expected nonempty folds and both stances")
        out_of_fold[held] = predict(x[held], fit(x[source], y[source]))
    limits = np.maximum(base.cutoff(out_of_fold, y), 0)
    final = fit(x, y)
    paths = {
        "single_pass_compact": lambda batch: predict(compact.features(batch), final),
        "two_pass_compact": lambda batch: cascade.contrast(base.small_scores(compact, batch)[0]),
        "ternary_4b": lambda batch: cascade.direct(tokenizer, large, batch),
    }
    cutoffs = {
        "single_pass_compact": limits.tolist(),
        "two_pass_compact": anchor["reports"]["contrastive_binary"]["cutoff"],
        "ternary_4b": anchor["reports"]["ternary_reader"]["cutoff"],
    }
    baseline_name = {"two_pass_compact": "contrastive_binary", "ternary_4b": "ternary_reader"}
    for op in paths.values():
        op(test[:cascade.BATCH])
    reports = {}
    order = tuple(paths)[::(-1 if args.order_offset else 1)]
    for arm in order:
        scores, times = timed(test, paths[arm])
        result = base.tally(scores, gold["test"], cutoffs[arm])
        if arm in baseline_name and result != anchor["reports"][baseline_name[arm]]["calibrated"]["test"]:
            raise ValueError(f"Baseline changed: {arm}")
        reports[arm] = dict(outcomes=result, cutoff=cutoffs[arm], p95_ms=float(np.quantile(times, .95)),
                            median_ms=float(np.median(times)))
        print(f"{arm}: {result['only_correct']} correct and {result['only_opposite']} opposite", flush=True)
    write_json(args.output, dict(
        reports=reports, calibration_out_of_fold=base.tally(out_of_fold, y, limits),
        folds=[len(fold) for fold in folds], feature_width=x.shape[1], ridge=RIDGE,
        order_offset=args.order_offset, gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        reference_sha256=digest(args.reference), projection_sha256=digest(args.projection),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "reranker", "chat", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
