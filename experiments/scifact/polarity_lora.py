"""Compare frozen and low-rank-adapted one-pass stance verification."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from expert_search.qwen_reranker import pack_tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.scifact import polarity_gate as base
from experiments.scifact.polarity_readout import TASK, partitions

RANK = 8
LAYERS = 4
EPOCHS = 4
LR = 0.0005
TRAIN_BATCH = 8
TEST_BATCH = 16
ARMS = ("head_only", "lora")


class LowRankProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.down = nn.Linear(original.in_features, RANK, bias=False, device="cuda")
        self.up = nn.Linear(RANK, original.out_features, bias=False, device="cuda")
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs):
        return self.original(inputs) + self.up(self.down(inputs.float())).to(inputs.dtype)


class TrainableStance:
    def __init__(self, checkpoint):
        self.reader = Reranker(checkpoint)
        self.model = self.reader.model.model.requires_grad_(False).eval()
        width = self.model.config.hidden_size
        self.head = nn.Linear(width, 2, device="cuda")
        self.layers = self.model.layers[-LAYERS:]
        self.original = [{name: getattr(layer.self_attn, name) for name in ("q_proj", "v_proj")}
                         for layer in self.layers]

    def reset(self, arm, seed):
        torch.manual_seed(seed)
        self.head.reset_parameters()
        for layer, projections in zip(self.layers, self.original, strict=True):
            for name, original in projections.items():
                member = LowRankProjection(original) if arm == "lora" else original
                setattr(layer.self_attn, name, member)

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.head.parameters())

    def encode(self, rows):
        sequences = []
        for row in rows:
            body = self.reader.tokenizer.encode(
                f"<Instruct>: {TASK}\n<Query>: {row['criterion']}\n<Document>: {row['source']}",
                add_special_tokens=False,
            )
            packed, full = pack_tokens(self.reader.prefix, body, self.reader.suffix, 1024)
            if len(packed) != full:
                raise ValueError("Unexpected truncated sequence")
            sequences.append(dict(input_ids=packed))
        return sequences

    def logits(self, sequences):
        inputs = self.reader.tokenizer.pad(sequences, padding=True, return_tensors="pt").to("cuda")
        terminal = self.model(**inputs, use_cache=False).last_hidden_state[:, -1, :].float()
        return self.head(torch.nn.functional.normalize(terminal) * (terminal.shape[1] ** 0.5))

    def train(self, sequences, target, rows, arm, seed):
        self.reset(arm, seed)
        params = self.params()
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(EPOCHS):
            epoch_loss = []
            order = rng.permutation(rows)
            for i in range(0, len(order), TRAIN_BATCH):
                batch = order[i:i + TRAIN_BATCH]
                optimizer.zero_grad(set_to_none=True)
                logits = self.logits([sequences[j] for j in batch])
                loss = nn.functional.cross_entropy(logits, target[batch.tolist()])
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_loss)))
        trained = sum(p.numel() for p in params)
        return trained, losses

    @torch.inference_mode()
    def predict(self, rows, *, raw=False):
        outcomes, times = [], []
        for i in range(0, len(rows), TEST_BATCH):
            begin = perf_counter()
            batch = rows[i:i + TEST_BATCH]
            sequences = self.encode(batch) if raw else batch
            scores = self.logits(sequences).cpu().numpy()
            margin = scores[:, 0] - scores[:, 1]
            outcomes.append(np.column_stack((margin, -margin)))
            times.append((perf_counter() - begin) * 1000)
        return np.concatenate(outcomes), times


def evaluate(model, panel, sequences, gold, folds, arm):
    target = torch.tensor(gold["calibration"], dtype=torch.long, device="cuda")
    oof = np.empty((len(target), 2))
    fit = []
    for i, held in enumerate(folds):
        rows = np.setdiff1d(np.arange(len(target)), held)
        _, curve = model.train(sequences, target, rows, arm, seed=112 + i)
        oof[held], _ = model.predict([sequences[j] for j in held])
        fit.append(curve)
    cutoffs = np.maximum(base.cutoff(oof, gold["calibration"]), 0)
    training_count, curve = model.train(sequences, target, np.arange(len(target)), arm, seed=116)
    model.predict(panel["test"][:TEST_BATCH], raw=True)
    margins, timings = model.predict(panel["test"], raw=True)
    return dict(
        trainable_parameters=training_count, fit_cross_entropy=curve, fold_cross_entropy=fit,
        cutoffs=cutoffs.tolist(), calibration=base.tally(oof, gold["calibration"], cutoffs),
        test=base.tally(margins, gold["test"], cutoffs),
        test_zero=base.tally(margins, gold["test"], [0, 0]),
        p95_ms=float(np.quantile(timings, .95)),
    )


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm GPU")
    anchor = json.loads(args.reference.read_bytes())
    if digest(args.projection) != anchor["projection_sha256"]:
        raise ValueError("Changed evidence projection")
    panel = json.loads(args.projection.read_bytes())["rows"]
    gold = {name: np.array([base.LABELS.index(r["label"]) for r in rows])
            for name, rows in panel.items()}
    model = TrainableStance(args.reranker)
    if model.reader.provenance["revision"] != anchor["reranker_revision"]:
        raise ValueError("Changed checkpoint")
    folds = partitions(panel["calibration"])
    sequences = model.encode(panel["calibration"])
    reports = {}
    for arm in (ARMS[::-1] if args.order_offset else ARMS):
        reports[arm] = evaluate(model, panel, sequences, gold, folds, arm)
        row = reports[arm]["test"]
        print(f"{arm}: completed training and test ({row['only_correct']} correct, {row['only_opposite']} opposite)", flush=True)
    write_json(args.output, dict(
        reports=reports, folds=[len(fold) for fold in folds], iterations=EPOCHS,
        rank=RANK, lora_layers=LAYERS, lr=LR, gpu=torch.cuda.get_device_name(),
        reference_sha256=digest(args.reference), projection_sha256=digest(args.projection),
        script_sha256=digest(Path(__file__)), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
