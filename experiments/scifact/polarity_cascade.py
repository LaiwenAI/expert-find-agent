"""Allocate an expensive scientific evidence reader within a batch budget."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.qasper.paired_reader import load_reader
from experiments.scifact import polarity_gate as base
from experiments.scifact.criterion_reader import measure

BATCH = 16
FRACTIONS = (0.25, 0.5)
RIDGE = 0.1


def contrast(raw):
    return raw - raw[:, ::-1]


def direct(tokenizer, reader, rows):
    logits, _ = measure(tokenizer, reader, rows)
    return logits[:, :2] - np.maximum(logits[:, 1::-1], logits[:, 2:])


def utilities(margins, gold, cutoff):
    accept = margins > cutoff
    i = np.arange(len(gold))
    return accept[i, gold].astype(float) - accept[i, 1 - gold].astype(float)


def design(raw):
    return np.column_stack([np.ones(len(raw)), raw, raw.min(axis=1), abs(raw[:, 0] - raw[:, 1])])


def fit(raw, utility):
    features = design(raw)
    penalty = np.eye(features.shape[1]) * RIDGE
    penalty[0, 0] = 0
    return np.linalg.solve(features.T @ features / len(raw) + penalty,
                           features.T @ utility / len(raw))


def priorities(raw, rows, weights):
    random = np.array([int.from_bytes(sha256(f"{r['claim_id']}:{r['document']}".encode()).digest()[:8], "big")
                       for r in rows], dtype=np.uint64)
    return {
        "hash_control": random,
        "uncertainty": -abs(raw[:, 0] - raw[:, 1]),
        "dual_confidence": raw.min(axis=1),
        "learned_value": design(raw) @ weights,
    }


def allocate(values, fraction):
    selected = np.zeros(len(values), dtype=bool)
    for start in range(0, len(values), BATCH):
        batch = values[start:start + BATCH]
        limit = int(len(batch) * fraction)
        if limit:
            order = sorted(range(len(batch)), key=lambda i: (-int(batch[i]) if batch.dtype.kind == "u"
                                                            else -float(batch[i]), i))
            selected[start + np.asarray(order[:limit])] = True
    return selected


def record(small, large, selected, gold):
    scores = np.where(selected[:, None], large, small)
    return dict(expensive_items=int(selected.sum()), measurements=base.tally(scores, gold, [0, 0]))


def live(compare, rows, tokenizer, reader, verifier, cutoffs, weights):
    timings, decisions = [], []
    calls = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        begin = perf_counter()
        if compare == "all_large":
            output = direct(tokenizer, reader, batch) - cutoffs[1]
            calls += len(batch)
        else:
            raw, _ = base.small_scores(verifier, batch)
            output = contrast(raw) - cutoffs[0]
            if compare != "all_small":
                selected = allocate(priorities(raw, batch, weights)[compare], 0.5)
                if selected.any():
                    routed = [row for row, take in zip(batch, selected, strict=True) if take]
                    output[selected] = direct(tokenizer, reader, routed) - cutoffs[1]
                    calls += len(routed)
        timings.append((perf_counter() - begin) * 1000)
        decisions.extend((output > 0).tolist())
    return np.asarray(decisions), calls, timings


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm GPU")
    anchor = json.loads(args.reference.read_bytes())
    if digest(args.projection) != anchor["projection_sha256"]:
        raise ValueError("Changed evidence projection")
    samples = json.loads(args.projection.read_bytes())["rows"]
    verifier = Reranker(args.reranker)
    tokenizer, reader, provenance = load_reader(args.chat)
    if verifier.provenance["revision"] != anchor["reranker_revision"] or provenance["revision"] != anchor["chat_revision"]:
        raise ValueError("Changed checkpoints")
    small_cut = np.array(anchor["reports"]["contrastive_binary"]["cutoff"])
    large_cut = np.array(anchor["reports"]["ternary_reader"]["cutoff"])
    gold = {key: np.array([base.LABELS.index(r["label"]) for r in rows]) for key, rows in samples.items()}
    raw, small, large = {}, {}, {}
    for key, rows in samples.items():
        raw[key], _ = base.small_scores(verifier, rows)
        small[key] = contrast(raw[key]) - small_cut
        large[key] = direct(tokenizer, reader, rows) - large_cut
        for alias, scores in (("contrastive_binary", small), ("ternary_reader", large)):
            counts = base.tally(scores[key], gold[key], [0, 0])
            if counts != anchor["reports"][alias]["calibrated"][key]:
                raise ValueError(f"Reference results changed: {key}/{alias}")
    y = gold["calibration"]
    gain = utilities(large["calibration"], y, [0, 0]) - utilities(small["calibration"], y, [0, 0])
    weights = fit(raw["calibration"], gain)
    priors = priorities(raw["test"], samples["test"], weights)
    report = {
        str(fraction): {name: record(small["test"], large["test"], allocate(priority, fraction), gold["test"])
                        for name, priority in priors.items()}
        for fraction in FRACTIONS
    }

    live_results = {}
    arms = ("all_small", "all_large", "uncertainty", "learned_value", "dual_confidence")
    if args.order_offset:
        arms = arms[::-1]
    for name in arms:
        choices, calls, times = live(name, samples["test"], tokenizer, reader, verifier,
                                     (small_cut, large_cut), weights)
        if name == "all_small":
            expected = small["test"] > 0
        elif name == "all_large":
            expected = large["test"] > 0
        else:
            expected = np.where(allocate(priors[name], 0.5)[:, None], large["test"], small["test"]) > 0
        live_results[name] = dict(
            calls=calls, decisions=base.tally(choices.astype(float), gold["test"], [0, 0]),
            changed_rows=int(np.any(choices != expected, axis=1).sum()),
            batch_p95_ms=float(np.quantile(times, .95)), batch_median_ms=float(np.median(times)),
        )
        print(f"{name}: {calls} actual 4B items", flush=True)
    write_json(args.output, dict(
        routing=report, live=live_results, batch_size=BATCH, ridge=RIDGE, order_offset=args.order_offset,
        gpu=torch.cuda.get_device_name(), provenance_sha256=digest(args.reference),
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "reranker", "chat", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
