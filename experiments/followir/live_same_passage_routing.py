"""Execute topic-local verifier routing one revision at a time."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir import same_passage_routing as study
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.evaluate_same_passage_routing import cutoff


ARMS = ("all_0.6b", "all_4b", "quarter_boundary")


def batch_score(reader, prompt, documents, *, compact):
    result = []
    for cursor in range(0, len(documents), 8):
        batch = documents[cursor:cursor + 8]
        if compact:
            scores, clipped = reader.score_pairs([(prompt, body) for body in batch])
        else:
            scores, clipped = reader.score(prompt, batch)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Unexpected live verifier output")
        result.extend(scores)
    return result


def execute(arm, small, large, rows, thresholds, choice_keys):
    _, _, _, query, (_, instruction), _ = rows[0]
    prompt = f"{query}\nRelevance criteria: {instruction}"
    documents = [item[5] for item in rows]
    began = perf_counter()
    if arm == "all_4b":
        scored = batch_score(large, prompt, documents, compact=False)
        approved = [value > thresholds[1] for value in scored]
        return approved, list(range(len(scored))), 0, (perf_counter() - began) * 1000

    scored = batch_score(small, prompt, documents, compact=True)
    approved = [value > thresholds[0] for value in scored]
    selected = []
    extra = 0
    if arm in ("quarter_boundary", "mixed_next_four"):
        ranked = sorted(range(len(scored)), key=lambda i: (
            abs(scored[i] - thresholds[0]), choice_keys[i],
        ))
        quota = len(scored) // 4
        selected = ranked[:quota]
        if arm == "mixed_next_four" and quota:
            from experiments.followir.criterion_contrast_routing import score_topic

            subset = ranked[quota - 1:quota + 3]
            votes = score_topic(small, query, [documents[i] for i in subset])
            extra = len(subset)
            selected[-1] = max(zip(votes, subset, strict=True), key=lambda pair: pair[0])[1]
        upgrades = batch_score(large, prompt, [documents[i] for i in selected], compact=False)
        for index, value in zip(selected, upgrades, strict=True):
            approved[index] = value > thresholds[1]
    return approved, selected, extra, (perf_counter() - began) * 1000


def run(args):
    from hashlib import sha256

    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and a Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads(args.scores.read_bytes())
    if source["source_sha256"] != study.gate.SNAPSHOT_HASHES:
        raise ValueError("Score replay has different inputs")
    panels = {key: val["rows"] for key, val in source["panels"].items()}
    selected_arms = ARMS
    if args.contrast:
        contrast = json.loads(args.contrast.read_bytes())
        if (contrast["prior_sha256"] != digest(args.scores)
                or contrast["checkpoint"] != source["compact_revision"]):
            raise ValueError("Mismatched topic-only replay")
        panels["robust"] = [dict(row, topic_only=margin) for row, margin in
                            zip(panels["robust"], contrast["margins"]["robust"], strict=True)]
        selected_arms = (*ARMS, "mixed_next_four")
    dev = {key: panels[key] for key in ("core", "news")}
    thresholds = tuple(cutoff(dev, name) for name in ("cheap", "costly"))
    sample = study.samples(args)["robust"]
    by_topic = defaultdict(list)
    for row, record in zip(sample, panels["robust"], strict=True):
        if (row[0], row[1], row[2]) != (record["topic"], record["label"], record["key"]):
            raise ValueError("Live row did not match calibration projection")
        by_topic[row[0]].append((row, record))

    small = study.Reranker(args.reranker)
    large = study.InstructionReader(args.chat)
    if small.provenance["revision"] != source["compact_revision"] or large.revision != source["strong_revision"]:
        raise ValueError("Live checkpoints differ")
    totals = {arm: Counter() for arm in selected_arms}
    times = defaultdict(list)

    def measure(topic_rows, index, warm=False):
        rows, recorded = zip(*topic_rows)
        keys = [sha256(f"same-passage-v1:{row[0]}:{row[2]}".encode()).digest() for row in rows]
        nearest = sorted(range(len(rows)), key=lambda i: (
            abs(recorded[i]["cheap"] - thresholds[0]), keys[i],
        ))[:len(rows) // 4]
        arms = selected_arms[index % len(selected_arms):] + selected_arms[:index % len(selected_arms)]
        if args.contrast:
            from experiments.followir.evaluate_criterion_contrast_routing import mixed_recovery

            planned, _ = mixed_recovery(
                list(range(len(rows))), [abs(r["cheap"] - thresholds[0]) for r in recorded],
                [r["topic_only"] for r in recorded], keys, len(rows) // 4, shortlist=4,
            )
        replay = {"all_0.6b": [], "all_4b": list(range(len(rows))), "quarter_boundary": nearest}
        if args.contrast:
            replay["mixed_next_four"] = planned
        for arm in arms:
            approved, selected, extra, latency = execute(arm, small, large, rows, thresholds, keys)
            if warm:
                continue
            times[arm].append(latency)
            chosen_set = set(replay[arm])
            if args.contrast:
                totals[arm]["extra_small_scores"] += extra
            expected = [r["costly"] > thresholds[1] if i in chosen_set else r["cheap"] > thresholds[0]
                        for i, r in enumerate(recorded)]
            totals[arm].update(upgrades=len(selected), decisions_changed=sum(a != b for a, b in zip(approved, expected)),
                               topic_routes_changed=int(set(selected) != chosen_set))
            for r, accepts in zip(rows, approved, strict=True):
                totals[arm][r[1]] += accepts

    first = next(iter(by_topic.values()))
    measure(first, 0, warm=True)
    sequence = list(by_topic.values())[::(-1 if args.order_offset else 1)]
    for i, group in enumerate(sequence):
        measure(group, i + args.order_offset)
    write_json(args.output, dict(
        arms={name: dict(totals[name], mean_ms=float(np.mean(times[name])),
                         p95_ms=float(np.quantile(times[name], 0.95))) for name in selected_arms},
        topics=len(by_topic), cutoffs=thresholds, gpu=torch.cuda.get_device_name(),
        score_sha256=digest(args.scores), code_sha256=digest(Path(__file__)), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "news", "robust", "reranker", "chat", "scores", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    cli.add_argument("--contrast", type=Path)
    run(cli.parse_args())
