"""Compare budgeted verifier choices on human-labeled criterion changes."""

import argparse
from collections import defaultdict
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.retrieval import tokens
from experiments.followir import paired_criterion_gate as gate
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.followir.rescue_reader_diagnosis import InstructionReader


def samples(paths):
    datasets = {name: gate.load(getattr(paths, name), expected)
                for name, expected in gate.SNAPSHOT_HASHES.items()}
    development = {name: gate.sample(data, data["pairs"])
                   for name, data in datasets.items() if name != "robust"}
    prior = {name: (datasets[name], None, None) for name in development}
    topics = gate.independent_topics(prior, datasets["robust"])
    excluded = {row[2] for rows in development.values() for row in rows}
    return dict(development, robust=gate.sample(datasets["robust"], topics, excluded))


def fraction(terms, document):
    return len(terms & document) / max(len(terms), 1)


def score_split(rows, small, large):
    compact, short_clips = gate.score(small, rows, gate.PROMPTS["relevance"])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[0]].append(index)
    strong = np.full(len(rows), np.nan)
    long_clips = 0
    for indices in groups.values():
        _, _, _, query, (_, instruction), _ = rows[indices[0]]
        assert all(rows[i][3:5] == rows[indices[0]][3:5] for i in indices)
        prompt = f"{query}\nRelevance criteria: {instruction}"
        for start in range(0, len(indices), 8):
            batch = indices[start:start + 8]
            margins, clipped = large.score(prompt, [rows[i][5] for i in batch])
            strong[batch] = margins
            long_clips += clipped
    if not np.isfinite(strong).all():
        raise ValueError("Invalid 4B margins")

    records = []
    for row, (_, cheap), costly in zip(rows, compact, strong, strict=True):
        topic, label, doc_hash, query, (_, instruction), text = row
        q_words, source = set(tokens(query)), set(tokens(text))
        criteria = set(tokens(instruction)) - q_words
        records.append(dict(topic=topic, label=label, key=doc_hash, cheap=float(cheap), costly=float(costly),
                            query_coverage=fraction(q_words, source), criterion_coverage=fraction(criteria, source),
                            length=len(text)))
    return records, {"0.6b": short_clips, "4b": long_clips}


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a new result file and a Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    panels = samples(args)
    small = Reranker(args.reranker)
    large = InstructionReader(args.chat)
    reports = {}
    order = list(panels)[::(-1 if args.order_offset else 1)]
    for name in order:
        rows, clips = score_split(panels[name], small, large)
        reports[name] = dict(rows=rows, clipped=clips)
        print(f"{name}: {len(rows)} assessed document–query pairs scored", flush=True)
    write_json(args.output, dict(
        panels=reports, gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
        compact_revision=small.provenance["revision"], strong_revision=large.revision,
        source_sha256=gate.SNAPSHOT_HASHES, code_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "news", "robust", "reranker", "chat", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(cli.parse_args())
