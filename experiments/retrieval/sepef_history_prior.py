"""Audit temporal SE-PEF scope and measure text-free historical expert priors."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
import json
import os
from pathlib import Path
from statistics import mean

from experiments.retrieval.sepef_temporal_panel import PREFIX, SNAPSHOT, ANSWERS_PER_EXPERT


FILES = ("expert_test_data.json", "val/qrels.json", "test/qrels.json")


def ranks(counter):
    return sorted(counter, key=lambda author: (-counter[author], author))[:10]


def scope(history, limit):
    activity = Counter()
    communities = defaultdict(Counter)
    for author, row in history.items():
        records = list(zip(row["answer_timestamps"], row["answers"], strict=True))
        if any(not isinstance(time, int) or time >= SNAPSHOT for time, _ in records):
            raise ValueError("History reaches or exceeds the snapshot boundary")
        if limit is not None:
            records = sorted(records, reverse=True)[:limit]
        for _, answer_id in records:
            activity[author] += 1
            community = answer_id.split("_", 1)[0]
            communities[community][author] += 1
    if sum(activity.values()) != sum(sum(row.values()) for row in communities.values()):
        raise AssertionError("Inconsistent community attribution")
    return activity, communities


def evaluate(activity, communities, qrels):
    global_order = ranks(activity)
    local_order = {site: ranks(counts) for site, counts in communities.items()}
    positives = {query: set(answers) for query, answers in qrels.items()}
    output = {}
    for local in (False, True):
        reciprocal, hits = [], []
        available = 0
        for query, labels in positives.items():
            site = query.split("_", 1)[0]
            candidates = communities.get(site, Counter()) if local else activity
            available += bool(labels & candidates.keys())
            returned = local_order.get(site, []) if local else global_order
            rank = next((i for i, author in enumerate(returned, 1) if author in labels), None)
            reciprocal.append(0.0 if rank is None else 1.0 / rank)
            hits.append(rank is not None and rank <= 5)
        output["community" if local else "global"] = dict(
            label_in_scope=available, hit5=mean(hits), mrr10=mean(reciprocal),
        )
    return output


def run(data, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require an unused result path and a Slurm compute node")
    inputs = {name: (data / name).read_bytes() for name in FILES}
    history = json.loads(inputs[FILES[0]])
    qrels = {split: json.loads(inputs[f"{split}/qrels.json"]) for split in ("val", "test")}
    if qrels["val"].keys() & qrels["test"].keys():
        raise ValueError("Released validation and test qrels overlap")
    owned = list(chain.from_iterable(row["answers"] for row in history.values()))
    if len(set(owned)) != len(owned):
        raise ValueError("Repeated history answer ID")
    reports = {}
    for name, limit in (("all_history", None), ("recent8", ANSWERS_PER_EXPERT)):
        activity, communities = scope(history, limit)
        reports[name] = dict(
            experts=len(activity), answers=sum(activity.values()), communities=len(communities),
            report={split: evaluate(activity, communities, labels) for split, labels in qrels.items()},
        )
    result = dict(
        job=os.environ["SLURM_JOB_ID"], upstream=PREFIX,
        source_sha256={name: sha256(raw).hexdigest() for name, raw in inputs.items()},
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        profile_cutoff_utc=SNAPSHOT, released_experts=len(history),
        latest_history_timestamp=max(t for row in history.values() for t in row["answer_timestamps"]),
        qrels={split: len(labels) for split, labels in qrels.items()}, reports=reports,
    )
    with output.open("x") as writer:
        json.dump(result, writer, indent=2, sort_keys=True)
        writer.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
