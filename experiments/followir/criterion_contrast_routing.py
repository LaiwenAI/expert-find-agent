"""Measure topic-only margins for criterion-contrast verification routing."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.followir import same_passage_routing as study
from experiments.followir.core17_reranking import digest, write_json


def score_topic(reader, query, documents):
    result = []
    for start in range(0, len(documents), 8):
        pairs = [(query, text) for text in documents[start:start + 8]]
        margins, clipped = reader.score_pairs(pairs, task=study.gate.PROMPTS["relevance"])
        if clipped or not np.isfinite(margins).all():
            raise ValueError("Unexpected topic-only verifier output")
        result.extend(margins)
    return result


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result and a Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    original = json.loads(args.scores.read_bytes())
    if original["source_sha256"] != study.gate.SNAPSHOT_HASHES:
        raise ValueError("Mismatched source projection")
    panels = study.samples(args)
    compact = study.Reranker(args.reranker)
    if compact.provenance["revision"] != original["compact_revision"]:
        raise ValueError("Changed reranker checkpoint")
    results = {}
    order = list(panels)[::(-1 if args.order_offset else 1)]
    for name in order:
        rows = panels[name]
        original_rows = original["panels"][name]["rows"]
        groups = defaultdict(list)
        for i, (row, orig) in enumerate(zip(rows, original_rows, strict=True)):
            if row[:3] != (orig["topic"], orig["label"], orig["key"]):
                raise ValueError("Changed row identity")
            groups[row[0]].append(i)
        scores = {}
        for indices in groups.values():
            queries = {rows[i][3] for i in indices}
            if len(queries) != 1:
                raise ValueError("A topic must have a consistent base query")
            sequence = [rows[i][5] for i in indices]
            for i, margin in zip(indices, score_topic(compact, queries.pop(), sequence), strict=True):
                scores[i] = margin
        results[name] = [scores[i] for i in range(len(rows))]
        print(f"{name}: checked {len(rows)} unchanged passages without the criterion", flush=True)
    write_json(args.output, dict(
        margins=results, checkpoint=compact.provenance["revision"], prior_sha256=digest(args.scores),
        code_sha256=digest(Path(__file__)), offset=args.order_offset, gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("core", "news", "robust", "reranker", "scores", "output"):
        cli.add_argument(key, type=Path)
    cli.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(cli.parse_args())
