"""Test portable verifier cutoffs on paired human relevance judgments."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker, TASK
from experiments.followir.robust04_rescue_risk import independent_topics

SNAPSHOT_HASHES = {
    "core": "9a70b11bc1e87b9ca91843b4222cd7fca4a405596ce30c0fd97fffbd4dd6e219",
    "news": "fd0f3d1b85234d5e72a7bbeb6b95125cbe90d5bfd526b32f30b7ccc8106bcec4",
    "robust": "bc8891378219d39d3ee850a5f5c06ffabace9a921ac5243b7bfd0da7c1a5926f",
}
CLASSES = {(True, True): "stable_positive", (True, False): "invalidated",
           (False, False): "stable_negative"}
NEGATIVE_CLASSES = ("invalidated", "stable_negative")
PROMPTS = {
    "relevance": TASK,
    "strict": ("Decide whether the document satisfies the query and the detailed relevance criteria. "
               "Treat explicit exclusions and requirements as binding. Answer no when the document "
               "does not establish a required condition or meets an exclusion."),
}
DOCUMENTS_PER_CLASS = 4
CHARACTER_WINDOW = 1800
CALIBRATION_RISK = 0.10


def load(path, expected):
    if digest(path) != expected:
        raise ValueError("Unexpected FollowIR snapshot")
    return json.loads(path.read_bytes())


def sample(dataset, topics, excluded=()):
    excluded = set(excluded)
    rows = []
    for topic in topics:
        old, revised = topic["qrels"]
        by_class = {kind: [] for kind in CLASSES.values()}
        for doc in topic["pool"]:
            if doc not in old or doc not in revised:
                continue
            kind = CLASSES[(old[doc] > 0, revised[doc] > 0)]
            text = dataset["docs"][doc]
            doc_hash = sha256(text.encode()).hexdigest()
            if doc_hash not in excluded:
                order = sha256(f"{topic['topic']}:{doc}".encode()).digest()
                by_class[kind].append((order, doc_hash, text))
        for kind, candidates in by_class.items():
            for _, doc_hash, text in sorted(candidates)[:DOCUMENTS_PER_CLASS]:
                rows.append((topic["topic"], kind, doc_hash, topic["query"],
                             topic["instructions"], text[:CHARACTER_WINDOW]))
    return rows


def score(reader, rows, task):
    pairs = [(f"{query}\nRelevance criteria: {instruction}", text)
             for _, _, _, query, instructions, text in rows for instruction in instructions]
    scores = []
    clipped = 0
    for start in range(0, len(pairs), 8):
        margins, count = reader.score_pairs(pairs[start:start + 8], task=task)
        scores.extend(margins)
        clipped += count
    values = np.asarray(scores).reshape(-1, 2)
    if len(rows) != len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid verifier output")
    return values, clipped


def operating(rows, values):
    groups = {}
    for kind in NEGATIVE_CLASSES:
        scores = sorted((new for row, (_, new) in zip(rows, values, strict=True) if row[1] == kind),
                        reverse=True)
        if not scores:
            raise ValueError("Missing calibration negatives")
        # With acceptance defined as score > cutoff, ties stay below the cap.
        groups[kind] = float(scores[int(CALIBRATION_RISK * len(scores))])
    return groups


def summarize(rows, values, cutoff):
    counts = {kind: Counter(total=0, accept_before=0, accept_after=0, accept_both=0,
                            reject_both=0, accept_then_reject=0, reject_then_accept=0)
              for kind in CLASSES.values()}
    for row, (old, new) in zip(rows, values, strict=True):
        stat = counts[row[1]]
        before, after = old > cutoff, new > cutoff
        stat.update(total=1, accept_before=int(before), accept_after=int(after),
                    accept_both=int(before and after), reject_both=int(not before and not after),
                    accept_then_reject=int(before and not after), reject_then_accept=int(not before and after))
    return counts


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm GPU")
    datasets = {name: load(getattr(args, name), check) for name, check in SNAPSHOT_HASHES.items()}
    development = {name: sample(datasets[name], datasets[name]["pairs"]) for name in ("core", "news")}
    prior = {name: (datasets[name], None, None) for name in development}
    test_topics = independent_topics(prior, datasets["robust"])
    dev_texts = {row[2] for rows in development.values() for row in rows}
    test = sample(datasets["robust"], test_topics, excluded=dev_texts)
    samples = {**development, "robust": test}
    reader = Reranker(args.reranker)
    arms = {}
    names = list(PROMPTS)
    if args.order_offset:
        names.reverse()
    for name in names:
        scored = {split: score(reader, rows, PROMPTS[name]) for split, rows in samples.items()}
        cutoffs = {split: operating(rows, scored[split][0]) for split, rows in development.items()}
        transferred = max(value for split in cutoffs.values() for value in split.values())
        results = {}
        for split, rows in samples.items():
            values, clipped = scored[split]
            results[split] = dict(rows=len(rows), topics=len({row[0] for row in rows}),
                                  clipped=clipped, decisions={
                                      key: summarize(rows, values, cutoff)
                                      for key, cutoff in (("minus_two", -2), ("zero", 0),
                                                          ("calibrated", transferred))
                                  })
        arms[name] = dict(calibration_cutoffs=cutoffs, transferred_cutoff=transferred, panels=results)
        print(f"{name}: completed scoring all three collections", flush=True)
    write_json(args.output, dict(
        arms=arms, order_offset=args.order_offset, gpu=torch.cuda.get_device_name(),
        revision=reader.provenance["revision"], source_sha256=SNAPSHOT_HASHES,
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "news", "robust", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
