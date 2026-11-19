"""Measure community coverage under a shared per-expert SE-PEF history budget."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean

from experiments.retrieval.sepef_history_prior import scope, evaluate
from experiments.retrieval.sepef_temporal_panel import SNAPSHOT


ASSETS = {
    "expert_test_data.json": "9e2856bb4bcf09dbe8e565c278b28da3737d915f2f7273a6919c51fad3e6ae51",
    "val/qrels.json": "bf2a6b1d42c2c22612374fcceee33be983d43611481e2efe5df726cc2dd2be78",
    "test/qrels.json": "701ff0d4e1e13f07051c65d99f39196f7de694fdb7d1b755582c4202507de2d7",
}
BUDGETS = (4, 8, 16)
MODES = ("recent", "reserve_recent_sites", "reserve_active_sites")


def site(answer_id):
    return answer_id.split("_", 1)[0]


def select(records, budget, mode):
    """Keep at most one reserved record for each selected community, then fill by recency."""
    recent = sorted(records, reverse=True)
    if mode == "recent":
        return recent[:budget]
    latest = {}
    counts = Counter()
    for record in recent:
        community = site(record[1])
        latest.setdefault(community, record)
        counts[community] += 1
    if mode == "reserve_active_sites":
        preferred = sorted(latest, key=lambda community: (counts[community], latest[community]), reverse=True)
    else:
        preferred = sorted(latest, key=lambda community: latest[community], reverse=True)
    reserved = [latest[community] for community in preferred[:budget]]
    present = set(reserved)
    return (reserved + [record for record in recent if record not in present])[:budget]


def mask_history(history, budget, mode):
    represented = defaultdict(set)
    source_count = 0
    records = {}
    for author, item in history.items():
        rows = list(zip(item["answer_timestamps"], item["answers"], strict=True))
        selected = select(rows, budget, mode)
        if len(selected) != min(len(rows), budget) or len(set(selected)) != len(selected):
            raise AssertionError("Unequal history budget or repeated answer")
        source_count += len(selected)
        if selected:
            records[author] = selected
        for _, answer in selected:
            represented[site(answer)].add(author)
    return represented, source_count, records


def run(data, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output path and Slurm")
    inputs = {name: (data / name).read_bytes() for name in ASSETS}
    if {name: sha256(blob).hexdigest() for name, blob in inputs.items()} != ASSETS:
        raise ValueError("Official archive members changed")
    history = json.loads(inputs["expert_test_data.json"])
    qrels = {split: json.loads(inputs[f"{split}/qrels.json"]) for split in ("val", "test")}
    if qrels["val"].keys() & qrels["test"].keys():
        raise ValueError("Evaluation splits overlap")
    activity, full = scope(history, None)
    reports = {}
    for budget in BUDGETS:
        reports[budget] = {}
        for mode in MODES:
            seen, answers, records = mask_history(history, budget, mode)
            priors = {community: Counter({author: full[community][author] for author in authors})
                      for community, authors in seen.items()}
            reports[budget][mode] = dict(
                answers=answers, expert_community_edges=sum(map(len, seen.values())),
                avg_communities_per_expert=mean(len({site(answer) for _, answer in rows})
                                               for rows in records.values()),
                splits={split: evaluate(activity, priors, judgments)["community"]
                        for split, judgments in qrels.items()},
            )
    with output.open("x") as file:
        json.dump(dict(job=os.environ["SLURM_JOB_ID"], script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                       sources=ASSETS, snapshot=SNAPSHOT, authors=len(activity), reports=reports,
                       qrels={split: len(rows) for split, rows in qrels.items()}), file, sort_keys=True, indent=2)
        file.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("data", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
