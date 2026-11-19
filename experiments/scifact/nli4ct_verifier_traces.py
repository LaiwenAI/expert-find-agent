"""Collect per-depth Qwen judgments to train a single-call verifier policy."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.scifact.budgeted_sentence_selection import digest
from experiments.scifact.nli4ct_adaptive_depth import MAX_ROUNDS, initial, make_task, next_action, update
from experiments.scifact.nli4ct_adaptive_reader import PLAN, measure
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE


SPLITS = ("train", "dev", "Gold_test")
CANDIDATES = ("initial", "one", "two", "three")
SECTIONS = ("Adverse Events", "Eligibility", "Intervention", "Results")


def cases(archive):
    with ZipFile(archive) as bundle:
        raw = {split: json.loads(bundle.read(f"Complete_dataset/{split}.json")) for split in SPLITS}
        held = {row[key] for part in SPLITS[1:] for row in raw[part].values()
                for key in ("Primary_id", "Secondary_id") if key in row}
        cache = {}
        for split, rows in raw.items():
            for key, row in sorted(rows.items()):
                trials = tuple(row[side] for side in ("Primary_id", "Secondary_id") if side in row)
                if split == "train" and held.intersection(trials):
                    continue
                section = row["Section_id"]
                for paper in trials:
                    if paper not in cache:
                        cache[paper] = json.loads(bundle.read(f"Complete_dataset/CT json/{paper}.json"))
                yield split, key, row["Statement"], section, row["Label"], tuple(
                    (paper, cache[paper][section]) for paper in trials)


def load_controller(frozen):
    if digest(frozen) != PLAN:
        raise ValueError("Changed evidence-expansion controller")
    data = json.loads(frozen.read_text())
    model = np.asarray(data["learned_coefficients"]), data["learned_intercept"], data["training_actions"]
    return model, data["results"]["learned"]["calibration"]["cutoff"]


def text_features(statement, section, sources, controller):
    task = make_task("", statement, sources, (frozenset(),) * len(sources))
    state = initial(task)
    views = [state]
    scores = []
    for _ in range(MAX_ROUNDS):
        action, score = next_action(task, state, controller, "learned")
        if action is not None:
            state = update(state, action)
        scores.append(0. if score is None else score)
        views.append(state)
    weights = task.weights
    query_mass = sum(weights.values()) or 1.
    static = [
        len(task.trials), len(statement.split()), len(weights), sum(t.isdigit() for t in weights),
        min(t.length for t in task.trials), max(t.length for t in task.trials),
        *(int(section == value) for value in SECTIONS), *scores,
    ]
    counts = []
    features = list(static)
    for indices in views:
        words = sum(len(lines[i].split()) for (_, lines), kept in zip(sources, indices, strict=True) for i in kept)
        counts.append(words)
        covered = [{term for i in kept for term in trial.terms[i]}
                   for trial, kept in zip(task.trials, indices, strict=True)]
        shares = [sum(weights[t] for t in sorted(terms)) / query_mass for terms in covered]
        features.extend([words, sum(map(len, indices)), min(shares), max(shares)])
    return views, scores, features, counts


def simulate(archive, frozen, checkpoint, output):
    import torch

    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh GPU allocation and output path")
    if digest(archive) != ARCHIVE:
        raise ValueError("Changed NLI4CT archive")
    controller, _ = load_controller(frozen)
    tokenizer, reader, provenance = load_reader(checkpoint)
    enc = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
    if any(len(ids) != 1 for ids in enc) or len({ids[0] for ids in enc}) != 3:
        raise ValueError("Require three distinct single-token digits")
    digits = [ids[0] for ids in enc]
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        metadata = dict(job=os.environ["SLURM_JOB_ID"], reader_revision=provenance["revision"],
                        gpu=torch.cuda.get_device_name(), archive_sha256=ARCHIVE,
                        plan_sha256=PLAN, script_sha256=digest(Path(__file__)))
        stream.write(json.dumps(metadata) + "\n")
        for serial, (split, key, statement, section, label, sources) in enumerate(cases(archive)):
            views, scores, features, words = text_features(statement, section, sources, controller)
            rotation = serial % len(CANDIDATES)
            results, memo = [None] * len(CANDIDATES), {}
            for depth in tuple(range(rotation, len(CANDIDATES))) + tuple(range(rotation)):
                view = views[depth]
                if view not in memo:
                    memo[view] = measure(tokenizer, reader, digits, statement, section, sources, view)
                results[depth] = memo[view]
            row = dict(split=split, key=key, trials=[p for p, _ in sources], scores=scores, xs=features, words=words,
                       target={"Entailment": 0, "Contradiction": 1}[label], readings=results)
            stream.write(json.dumps(row) + "\n")
            counts[split] += 1
            if (serial + 1) % 200 == 0:
                print(dict(counts), flush=True)
    if counts != {"train":1200, "dev":200, "Gold_test":500}:
        raise ValueError("Changed trial-blocked cohort")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "frozen", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    simulate(**vars(parser.parse_args()))
