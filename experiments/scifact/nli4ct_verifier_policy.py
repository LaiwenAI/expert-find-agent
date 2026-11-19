"""Choose evidence depth from anticipated verifier accuracy with one reader call."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.scifact.budgeted_sentence_selection import components, digest
from experiments.scifact.nli4ct_adaptive_reader import PLAN, measure
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE
from experiments.scifact.nli4ct_verifier_traces import CANDIDATES, cases, load_controller, text_features


BASE = 2
RIDGE = 10.
PRICE_GRID = np.r_[0., np.geomspace(.001, 1., 49)]
DEV_ALLOWED_LOSS = .01
TEST_ARMS = ("initial", "fixed_two", "fixed_three", "coverage_stop", "verifier_policy")


def read_traces(path):
    with path.open(encoding="utf-8") as stream:
        meta = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    cohort = Counter(row["split"] for row in rows)
    if (meta["archive_sha256"] != ARCHIVE or meta["plan_sha256"] != PLAN
            or cohort != {"train": 1200, "dev": 200, "Gold_test": 500}
            or len({r["key"] for r in rows}) != len(rows)):
        raise ValueError("Unexpected trajectory provenance or cohorts")
    sources = {split: {p for r in rows if r["split"] == split for p in r["trials"]} for split in cohort}
    if sources["train"] & (sources["dev"] | sources["Gold_test"]) or sources["dev"] & sources["Gold_test"]:
        raise ValueError("Trial overlap between evaluation and training")
    return meta, rows


def arrays(rows):
    matrix = lambda name: np.asarray([r[name] for r in rows], dtype=float)
    labels = np.asarray([r["target"] for r in rows])
    readings = np.asarray([r["readings"] for r in rows], dtype=int)
    return matrix("xs"), matrix("words"), readings[:, :, 0] == labels[:, None], readings[:, :, 2]


def fit(rows):
    features, _, correct, _ = arrays(rows)
    y = correct.astype(float) - correct[:, BASE, None]
    mean = features.mean(axis=0)
    scale = np.maximum(.01, features.std(axis=0))
    z = (features - mean) / scale
    offsets = y.mean(axis=0)
    fitted = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - offsets))
    weights = fitted / scale[:, None]
    return weights, offsets - mean @ weights


def policy_depth(features, words, weights, bias, price):
    cost = (words - words[..., BASE, None]) / 100.
    return (features @ weights + bias - price * cost).argmax(axis=-1)


def measure_choices(correct, tokens, choices):
    indices = np.arange(len(choices))
    return dict(correct=int(correct[indices, choices].sum()),
                mean_tokens=float(tokens[indices, choices].mean()),
                depths={CANDIDATES[k]: int(v) for k, v in sorted(Counter(choices.tolist()).items())})


def calibrate(rows, weights, bias):
    features, words, correct, tokens = arrays(rows)
    required = int(correct[:, BASE].sum()) - int(len(rows) * DEV_ALLOWED_LOSS)
    frontier = []
    for price in PRICE_GRID:
        choices = policy_depth(features, words, weights, bias, price)
        frontier.append(dict(price=float(price), **measure_choices(correct, tokens, choices)))
    fallback = dict(price=None, **measure_choices(correct, tokens, np.full(len(rows), BASE)))
    eligible = [row for row in [*frontier, fallback] if row["correct"] >= required]
    winner = min(eligible, key=lambda item: (item["mean_tokens"], -item["correct"]))
    return dict(required_correct=required, fixed_two=fallback, selected=winner, frontier=frontier)


def chosen_depth(xs, words, weights, bias, price):
    if price is None:
        return BASE
    return int(policy_depth(np.asarray(xs), np.asarray(words), weights, bias, price))


def coverage_depth(scores, cutoff):
    for depth, score in enumerate(scores):
        if score < cutoff:
            return depth
    return len(scores)


def row_groups(rows):
    edges = [dict(case=row["key"], doc=trial) for row in rows for trial in row["trials"]]
    mapping = {row["case"]: group for row, group in zip(edges, components(edges), strict=True)}
    return [mapping[row["key"]] for row in rows]


def compare_replay(rows, weights, bias, price, cutoff):
    features, words, correct, tokens = arrays(rows)
    ids = np.arange(len(rows))
    learned = (np.full(len(rows), BASE) if price is None else policy_depth(features, words, weights, bias, price))
    alternatives = {name: np.full(len(rows), depth) for depth, name in
                    enumerate(("initial", "fixed_one", "fixed_two", "fixed_three"))}
    alternatives["coverage_stop"] = np.asarray([coverage_depth(row["scores"], cutoff) for row in rows])
    groups = row_groups(rows)
    return dict(
        arms={name: measure_choices(correct, tokens, choices)
              for name, choices in alternatives.items()} | {"verifier_policy": measure_choices(correct, tokens, learned)},
        paired={name: clustered_paired_interval(correct[ids, learned].astype(int)-correct[ids, choices], groups)
                for name, choices in alternatives.items()},
        oracle_no_fixed_two_regression_mean_tokens=float(
            np.where(~correct[:, BASE, None] | correct, tokens, tokens.max()+1).min(axis=1).mean()),
        oracle_correct=int(correct.any(axis=1).sum()),
    )


def train(trajectories, frozen, output):
    _, cutoff = load_controller(frozen)
    meta, rows = read_traces(trajectories)
    weights, bias = fit([r for r in rows if r["split"] == "train"])
    dev = calibrate([r for r in rows if r["split"] == "dev"], weights, bias)
    price = dev["selected"]["price"]
    held = compare_replay([r for r in rows if r["split"] == "Gold_test"], weights, bias, price, cutoff)
    report = dict(
        archive_sha256=ARCHIVE, frozen_controller_sha256=PLAN, script_sha256=digest(Path(__file__)),
        trajectories_sha256=digest(trajectories), trajectory_meta=meta, ridge=RIDGE,
        coefficients=weights.tolist(), intercept=bias.tolist(), development=dev, replay=held,
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def arm_depth(arm, words, scores, features, weights, bias, price, cutoff):
    if arm == "coverage_stop":
        return coverage_depth(scores, cutoff)
    if arm == "verifier_policy":
        return chosen_depth(features, words, weights, bias, price)
    return {"initial":0, "fixed_two":2, "fixed_three":3}[arm]


def live(archive, frozen, result, checkpoint, output):
    import torch

    from experiments.followir.core17_reranking import write_json
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.nli4ct_adaptive_reader import choice

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output on a Slurm GPU node")
    if digest(archive) != ARCHIVE:
        raise ValueError("Changed archive")
    controller, cutoff = load_controller(frozen)
    prior = json.loads(result.read_text())
    if (prior["archive_sha256"] != ARCHIVE or prior["frozen_controller_sha256"] != PLAN
            or prior["script_sha256"] != digest(Path(__file__))):
        raise ValueError("Changed policy training provenance")
    routing = np.asarray(prior["coefficients"]), np.asarray(prior["intercept"]), prior["development"]["selected"]["price"]
    tokenizer, reader, provenance = load_reader(checkpoint)
    digits = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
    if any(len(ids) != 1 for ids in digits) or len({ids[0] for ids in digits}) != 3:
        raise ValueError("Require distinct single-token label digits")
    digits = [ids[0] for ids in digits]
    held = [row for row in cases(archive) if row[0] == "Gold_test"]
    def evaluate(row, arm):
        _, key, statement, section, label, sources = row
        start = perf_counter()
        if arm == "verifier_policy":
            views, scores, features, words = text_features(statement, section, sources, controller)
            depth = arm_depth(arm, words, scores, features, *routing, cutoff)
            selected = views[depth]
        else:
            original = {"initial":"initial", "fixed_two":"fixed_two", "fixed_three":"fixed_three",
                        "coverage_stop":"adaptive"}[arm]
            selected = choice(key, statement, sources, controller, cutoff, original)
        prediction, _, n_tokens = measure(tokenizer, reader, digits, statement, section, sources, selected)
        elapsed = (perf_counter()-start)*1000
        return dict(ok=prediction == {"Entailment":0, "Contradiction":1}[label], tokens=n_tokens, time_ms=elapsed)

    for arm in TEST_ARMS:
        evaluate(held[0], arm)
    trials = [dict(key=row[1], trials=[paper for paper, _ in row[5]]) for row in held]
    groups = row_groups(trials)
    observed = defaultdict(list)
    for idx, row in enumerate(held):
        rotation = idx % len(TEST_ARMS)
        for arm in TEST_ARMS[rotation:] + TEST_ARMS[:rotation]:
            observed[arm].append(evaluate(row, arm))
        if (idx + 1) % 100 == 0:
            print(idx+1, "complete", flush=True)
    base = np.asarray([row["ok"] for row in observed["fixed_two"]], dtype=int)
    observations = {}
    for arm, rows in observed.items():
        correct = np.asarray([row["ok"] for row in rows], dtype=int)
        if int(correct.sum()) != prior["replay"]["arms"][arm]["correct"]:
            raise ValueError(f"Replay and live changed correctness for {arm}")
        observations[arm] = dict(
            correct=int(correct.sum()), paired_vs_fixed_two=clustered_paired_interval(correct-base, groups),
            mean_tokens=float(np.mean([row["tokens"] for row in rows])),
            latency_p50_p95_ms=np.quantile([row["time_ms"] for row in rows], [.5, .95]).tolist(),
        )
    write_json(output, dict(
        policy_sha256=digest(result), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        reader_revision=provenance["revision"], script_sha256=digest(Path(__file__)), cases=len(held),
        arms=observations,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(required=True)
    for action, names in (
        (train, ("trajectories", "frozen", "output")),
        (live, ("archive", "frozen", "result", "checkpoint", "output")),
    ):
        sub = actions.add_parser(action.__name__)
        for name in names:
            sub.add_argument(name, type=Path)
        sub.set_defaults(run=action)
    parsed = vars(parser.parse_args())
    parsed.pop("run")(**parsed)
