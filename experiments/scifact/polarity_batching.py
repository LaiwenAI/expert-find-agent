"""Measure decision-preserving length-aware scientific verification batches."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.qasper.paired_reader import load_reader
from experiments.scifact import polarity_gate as base
from experiments.scifact.criterion_reader import BATCH, SYSTEM

WINDOW = 2 * BATCH
ARMS = ("fifo", "length_sorted", "guarded_length_sorted")


def encode(tokenizer, rows):
    output = []
    for row in rows:
        content = json.dumps({"claim": row["criterion"], "evidence": row["source"]}, ensure_ascii=False)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(ids) > 1024:
            raise ValueError("Unexpected context length")
        output.append(dict(input_ids=ids))
    return output


class Inference:
    def __init__(self, tokenizer, model, cutoff):
        self.tokenizer = tokenizer
        self.model = model
        self.cutoff = cutoff
        tokens = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
        if any(len(x) != 1 for x in tokens) or len({x[0] for x in tokens}) != 3:
            raise ValueError("Expected three label tokens")
        self.labels = [x[0] for x in tokens]

    @torch.inference_mode()
    def score(self, sequences):
        encoded = self.tokenizer.pad(sequences, padding=True, padding_side="left", return_tensors="pt").to("cuda")
        output = self.model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, self.labels]
        scores = output.float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite verification scores")
        return scores[:, :2] - np.maximum(scores[:, 1::-1], scores[:, 2:]) - self.cutoff, encoded.input_ids.numel()

    def window(self, sequences, arm, guard):
        fifo = np.arange(len(sequences))
        order = (fifo if arm == "fifo" else
                 sorted(fifo, key=lambda i: (len(sequences[i]["input_ids"]), i)))
        margins = np.empty((len(sequences), 2))
        tokens, rechecks = 0, 0
        for start in range(0, len(order), BATCH):
            indices = order[start:start + BATCH]
            if len(indices) == len(sequences):
                indices = fifo
            values, work = self.score([sequences[i] for i in indices])
            margins[indices] = values
            tokens += work
        if arm == "guarded_length_sorted" and len(sequences) > BATCH:
            for start in range(0, len(fifo), BATCH):
                indices = fifo[start:start + BATCH]
                if np.any(np.abs(margins[indices]) <= guard):
                    values, work = self.score([sequences[i] for i in indices])
                    margins[indices] = values
                    tokens += work
                    rechecks += 1
        return margins, tokens, rechecks


def execute(reader, rows, arm, guard=0):
    outputs, elapsed = [], []
    tokens, rechecks = 0, 0
    for start in range(0, len(rows), WINDOW):
        started = perf_counter()
        sequences = encode(reader.tokenizer, rows[start:start + WINDOW])
        values, work, extra = reader.window(sequences, arm, guard)
        outputs.append(values)
        tokens += work
        rechecks += extra
        elapsed.append((perf_counter() - started) * 1000)
    return np.concatenate(outputs), tokens, rechecks, elapsed


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm GPU")
    reference = json.loads(args.reference.read_bytes())
    if digest(args.projection) != reference["projection_sha256"]:
        raise ValueError("Changed evidence projection")
    rows = json.loads(args.projection.read_bytes())["rows"]
    tokenizer, model, provenance = load_reader(args.chat)
    if provenance["revision"] != reference["chat_revision"]:
        raise ValueError("Changed reader checkpoint")
    anchor = reference["reports"]["ternary_reader"]
    reader = Inference(tokenizer, model, np.array(anchor["cutoff"]))
    baseline, _, _, _ = execute(reader, rows["calibration"], "fifo")
    grouped, _, _, _ = execute(reader, rows["calibration"], "length_sorted")
    guard = float(np.abs(baseline - grouped).max())
    gold = np.asarray([base.LABELS.index(r["label"]) for r in rows["test"]])
    arms = ARMS[::-1] if args.order_offset else ARMS
    runs = {arm: execute(reader, rows["test"], arm, guard) for arm in arms}
    decisions = runs["fifo"][0] > 0
    if base.tally(runs["fifo"][0], gold, [0, 0]) != anchor["calibrated"]["test"]:
        raise ValueError("FIFO did not match the reference")
    reports = {}
    for arm, (margins, tokens, rechecks, elapsed) in runs.items():
        reports[arm] = dict(
            outcomes=base.tally(margins, gold, [0, 0]),
            changed_rows=int(np.any((margins > 0) != decisions, axis=1).sum()),
            changed_score_rows=int(np.any(margins != runs["fifo"][0], axis=1).sum()),
            executed_padded_tokens=tokens, fallback_original_batches=rechecks,
            total_ms=sum(elapsed), window_p95_ms=float(np.quantile(elapsed, 0.95)),
        )
        print(f"{arm}: {tokens} padded tokens, {rechecks} original batches replayed", flush=True)
    write_json(args.output, dict(
        reports=reports, guard=guard, window_size=WINDOW, forward_batch=BATCH,
        test_windows=len(runs["fifo"][3]), order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.reference),
        projection_sha256=digest(args.projection), gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "chat", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
