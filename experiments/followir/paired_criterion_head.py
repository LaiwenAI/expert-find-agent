"""Train a one-pass criterion verifier on frozen final-token features."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_reranker import pack_tokens
from experiments.followir import paired_criterion_gate as gate
from experiments.followir.core17_reranking import digest, write_json

RIDGE = 0.001
PAIR_WEIGHT = 1.0


class FrozenFeatures(gate.Reranker):
    @torch.inference_mode()
    def features(self, rows):
        pairs = [(f"{query}\nRelevance criteria: {criterion}", text)
                 for _, _, _, query, criteria, text in rows for criterion in criteria]
        vectors, scores = [], []
        for start in range(0, len(pairs), 8):
            sequences = []
            for query, text in pairs[start:start + 8]:
                body = self.tokenizer.encode(
                    f"<Instruct>: {gate.TASK}\n<Query>: {query}\n<Document>: {text}",
                    add_special_tokens=False,
                )
                packed, _ = pack_tokens(self.prefix, body, self.suffix, 1024)
                sequences.append(dict(input_ids=packed))
            inputs = self.tokenizer.pad(sequences, padding=True, return_tensors="pt").to("cuda")
            hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1:, :]
            logits = self.model.lm_head(hidden)[:, 0, self.labels].float()
            scores.append((logits[:, 0] - logits[:, 1]).cpu().numpy())
            vectors.append(torch.nn.functional.normalize(hidden[:, 0, :].float()).cpu().numpy())
        vectors = np.concatenate(vectors).reshape(len(rows), 2, -1)
        scores = np.concatenate(scores).reshape(len(rows), 2)
        if not np.isfinite(vectors).all() or not np.isfinite(scores).all():
            raise ValueError("Nonfinite features or scores")
        return vectors, scores


def fit(rows, features, pair_weight):
    labels = {kind: views for views, kind in gate.CLASSES.items()}
    target = np.asarray([labels[row[1]] for row in rows], dtype=float) * 2 - 1
    width = features.shape[-1]
    x = features.reshape(-1, width).astype(np.float64)
    y = target.reshape(-1)
    center = x.mean(axis=0)
    intercept = float(y.mean())
    x -= center
    matrix = x.T @ x / len(x) + np.eye(width) * RIDGE
    rhs = x.T @ (y - intercept) / len(x)
    if pair_weight:
        # Same document, edited criterion. A stable pair targets zero
        # movement, whereas a human-invalidated pair targets a fall.
        delta = (features[:, 1] - features[:, 0]).astype(np.float64)
        expected = target[:, 1] - target[:, 0]
        matrix += pair_weight * (delta.T @ delta / len(rows))
        rhs += pair_weight * (delta.T @ expected / len(rows))
    weights = np.linalg.solve(matrix, rhs)
    return weights, intercept - float(center @ weights)


def apply(features, model):
    weights, intercept = model
    return features @ weights + intercept


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm GPU")
    anchor = json.loads(args.reference.read_bytes())
    if anchor["source_sha256"] != gate.SNAPSHOT_HASHES:
        raise ValueError("Changed benchmark sources")
    datasets = {name: gate.load(getattr(args, name), check) for name, check in gate.SNAPSHOT_HASHES.items()}
    names = ("core", "news")
    samples = {name: gate.sample(datasets[name], datasets[name]["pairs"]) for name in names}
    topics = gate.independent_topics({name: (datasets[name], None, None) for name in names}, datasets["robust"])
    samples["robust"] = gate.sample(datasets["robust"], topics,
                                    excluded={row[2] for name in names for row in samples[name]})
    model = FrozenFeatures(args.reranker)
    if model.provenance["revision"] != anchor["revision"]:
        raise ValueError("Changed benchmark verifier")
    order = tuple(samples)
    if args.order_offset:
        order = order[::-1]
    features, base = {}, {}
    for name in order:
        features[name], base[name] = model.features(samples[name])
        expected = anchor["arms"]["relevance"]["panels"][name]["decisions"]
        for key, cutoff in (("zero", 0), ("minus_two", -2)):
            if gate.summarize(samples[name], base[name], cutoff) != expected[key]:
                raise ValueError(f"Original decisions did not reproduce on {name}/{key}")
        print(f"{name}: computed {len(samples[name])} frozen feature pairs", flush=True)

    all_dev = [row for name in names for row in samples[name]]
    dev_features = np.concatenate([features[name] for name in names])
    routes = {"verbalizer": base}
    for route, weight in (("pointwise_head", 0.0), ("paired_head", PAIR_WEIGHT)):
        # Development scores are out of domain. The cutoff sees neither
        # in-sample fitted scores nor labels from the Robust04 panel.
        bank = {name: apply(features[name], fit(samples[other], features[other], weight))
                for name, other in (names, names[::-1])}
        bank["robust"] = apply(features["robust"], fit(all_dev, dev_features, weight))
        routes[route] = bank

    reports = {}
    for route, scores in routes.items():
        limits = {name: gate.operating(samples[name], scores[name]) for name in names}
        cutoff = max(value for panel in limits.values() for value in panel.values())
        counts = {name: gate.summarize(samples[name], scores[name], cutoff) for name in samples}
        reports[route] = dict(calibration=limits, threshold=cutoff, panels=counts)
    write_json(args.output, dict(
        reports=reports, feature_width=dev_features.shape[-1], ridge=RIDGE, pair_weight=PAIR_WEIGHT,
        seedless=True, gpu=torch.cuda.get_device_name(), revision=model.provenance["revision"],
        order_offset=args.order_offset, source_sha256=gate.SNAPSHOT_HASHES,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "news", "robust", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
