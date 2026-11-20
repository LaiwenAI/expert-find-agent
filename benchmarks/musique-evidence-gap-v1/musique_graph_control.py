"""Compare linked reads against dense retrieval and diffusion on the same links."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from musique_search import (
    DEPTHS, Lexical, QwenTorchEmbedding, candidate_ranks, choices, document,
    link_index, load_pairs, pct,
)
from expert_search.statistics import paired_interval


def diffusion_edges(links):
    starts, targets, weights = [], [], []
    for index, neighbors in enumerate(links):
        destinations = sorted(neighbors) if neighbors else [index]
        starts.extend([index] * len(destinations))
        targets.extend(destinations)
        weights.extend([1 / len(destinations)] * len(destinations))
    return np.asarray(starts), np.asarray(targets), np.asarray(weights)


def diffuse(order, edges):
    starts, targets, weights = edges
    prior = np.zeros(len(order))
    prior[order[:16]] = 1 / np.arange(1, 17)
    prior /= prior.sum()
    value = prior
    for _ in range(3):
        propagated = np.bincount(targets, weights=weights * value[starts], minlength=len(order))
        value = 0.4 * prior + 0.6 * propagated
    return np.argsort(-value, kind="stable")[:max(DEPTHS)].tolist()


def main(args):
    pairs = load_pairs(args.archive)[:464]
    bank = sorted({document(p) for pos, _, _ in pairs for p in pos["paragraphs"]})
    bank_index = {doc: i for i, doc in enumerate(bank)}
    titles = [row[0] for row in bank]
    encoder = QwenTorchEmbedding(args.embedding, max_length=256, batch_size=64)
    vectors = torch.from_numpy(encoder.encode([title + "\n" + text for title, text in bank])).cuda()
    lexical = Lexical(bank)
    links = link_index(bank)
    graph = diffusion_edges(links)
    methods = ("dense", "hybrid", "linked", "diffusion")
    runs = {method: [] for method in methods}
    times = {method: [] for method in methods}
    hop = []
    for index, (pos, _, gold) in enumerate(pairs[80:]):
        ranks = candidate_ranks(pos["question"], encoder, vectors, lexical)
        expected = {bank_index[doc] for doc in gold}
        hop.append(len(expected))
        for offset in range(len(methods)):
            name = methods[(index + offset) % len(methods)]
            started = perf_counter()
            picked = (diffuse(ranks["hybrid"], graph) if name == "diffusion"
                      else choices(name, ranks, titles, links, len(bank), max(DEPTHS)))
            duration_ms = 1000 * (perf_counter() - started)
            runs[name].append([expected <= set(picked[:depth]) for depth in DEPTHS])
            times[name].append(duration_ms)

    summary = {"graph_nonzero_entries": len(graph[0]), "queries": len(hop), "by_depth": {},
               "policy_ms": {name: pct(values) for name, values in times.items()}}
    for column, depth in enumerate(DEPTHS):
        result = {}
        proposed = np.array(runs["linked"], dtype=np.int8)[:, column]
        for name, rows in runs.items():
            observed = np.array(rows, dtype=np.int8)[:, column]
            item = {"complete": int(observed.sum()),
                    "complete_by_hop": {str(n): sum(bool(v) and size == n for v, size in zip(observed, hop))
                                        for n in (2, 3, 4)}}
            if name != "linked":
                diffs = proposed.astype(float) - observed
                interval = paired_interval(diffs)
                item["linked_minus_this"] = {
                    "gain": int((diffs > 0).sum()), "loss": int((diffs < 0).sum()),
                    "paired_bootstrap": interval,
                }
            result[name] = item
        summary["by_depth"][str(depth)] = result
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "embedding", "output"):
        cli.add_argument(arg, type=Path)
    main(cli.parse_args())
