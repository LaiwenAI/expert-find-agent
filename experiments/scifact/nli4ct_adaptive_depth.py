"""Benchmark offline-trained stopping and expansion for given trial sections."""

import argparse
from dataclasses import dataclass
import json
from itertools import combinations
import math
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from expert_search.evidence_windows import choose_windows
from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.scifact.budgeted_sentence_selection import components, digest
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE


MAX_ROUNDS = 3
BLOCK = 3
RIDGE = 10.
CONTROLLERS = ("lexical", "learned")


@dataclass(frozen=True)
class Trial:
    paper: str
    length: int
    terms: tuple
    gold: frozenset
    initial: frozenset


@dataclass(frozen=True)
class Task:
    key: str
    trials: tuple
    weights: dict


def read_splits(archive):
    with ZipFile(archive) as source:
        trial_cache = {}
        splits = {}
        for split in ("train", "dev", "Gold_test"):
            raw = json.loads(source.read(f"Complete_dataset/{split}.json"))
            tasks = []
            for key, row in sorted(raw.items()):
                query = set(tokens(row["Statement"]))
                trials = []
                for side in ("Primary", "Secondary"):
                    paper = row.get(f"{side}_id")
                    if not paper:
                        continue
                    if paper not in trial_cache:
                        trial_cache[paper] = json.loads(source.read(f"Complete_dataset/CT json/{paper}.json"))
                    lines = trial_cache[paper][row["Section_id"]]
                    gold = frozenset(row[f"{side}_evidence_index"])
                    if not gold or not 0 <= min(gold) <= max(gold) < len(lines):
                        raise ValueError("Unexpected annotation")
                    _, initial = choose_windows(row["Statement"], lines)
                    terms = tuple(frozenset(tokens(line)) & query for line in lines)
                    trials.append(Trial(paper, len(lines), terms, gold, frozenset(initial)))
                all_lines = [terms for trial in trials for terms in trial.terms]
                weights = {term: math.log1p(len(all_lines) / (1 + sum(term in terms for terms in all_lines)))
                           for term in sorted(query)}
                tasks.append(Task(key, tuple(trials), weights))
            splits[split] = tasks
    held_trials = {trial.paper for split in ("dev", "Gold_test") for task in splits[split] for trial in task.trials}
    splits["train"] = [task for task in splits["train"] if all(t.paper not in held_trials for t in task.trials)]
    return splits


def actions(task, state):
    weights = task.weights
    total = math.fsum(weights.values()) or 1.
    digits = {term for term in weights if term.isdigit()}
    for side, (trial, selected) in enumerate(zip(task.trials, state, strict=True)):
        unread = sorted(set(range(trial.length)) - selected)
        if not unread:
            continue
        known = set().union(*(trial.terms[j] for j in sorted(selected)))
        width = min(len(unread), BLOCK)
        for start in range(len(unread) - width + 1):
            indices = tuple(unread[start:start+width])
            words = set().union(*(trial.terms[j] for j in indices))
            overlap = math.fsum(weights[t] for t in sorted(words)) / total
            new = math.fsum(weights[t] for t in sorted(words - known)) / total
            gap = min(abs(i-j) for i in indices for j in selected)
            features = (
                overlap, new, len(words & digits) / max(1, len(digits)),
                width / BLOCK, indices[0] / trial.length, (indices[-1]-indices[0]+1) / trial.length,
                1 / gap, len(selected) / trial.length, math.log1p(len(unread)),
                math.fsum(weights[t] for t in sorted(known)) / total, len(task.trials) - 1,
                gap == 1, overlap * (gap == 1), overlap * len(selected) / trial.length,
            )
            # The lexical control gives precedence to newly covered terms and breaks ties by locality.
            rank = (new, overlap, -gap, -side, -indices[0])
            yield side, indices, features, rank


def initial(task):
    return tuple(trial.initial for trial in task.trials)


def update(state, action):
    side, indices, _, _ = action
    return state[:side] + (state[side] | frozenset(indices),) + state[side+1:]


def fit(tasks):
    features, targets = [], []
    for task in tasks:
        state = initial(task)
        for _ in range(MAX_ROUNDS):
            options = list(actions(task, state))
            if not options:
                break
            for side, indices, xs, _ in options:
                gold = task.trials[side].gold
                before = gold - state[side]
                gain = len(before.intersection(indices)) / len(gold)
                finishes = before and before.issubset(indices)
                targets.append((gain + int(bool(finishes))) / len(indices))
                features.append(xs)
            state = update(state, max(options, key=lambda item: item[3]))
    x = np.asarray(features)
    mean, scale = x.mean(axis=0), np.maximum(.01, x.std(axis=0))
    z = (x - mean) / scale
    y = np.asarray(targets)
    coef = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    weights = coef / scale
    return weights, float(y.mean() - weights @ mean), len(y)


def build_paths(task, model):
    weights, bias, _ = model
    routes = {}
    for mode in CONTROLLERS:
        started = perf_counter()
        state = initial(task)
        states, scores = [state], []
        for _ in range(MAX_ROUNDS):
            options = list(actions(task, state))
            if not options:
                break
            if mode == "learned":
                values = np.asarray([option[2] for option in options]) @ weights + bias
                choice = int(np.argmax(values))
                scores.append(float(values[choice]))
            else:
                choice = max(range(len(options)), key=lambda i: options[i][3])
                new, overlap, proximity, _, _ = options[choice][3]
                scores.append(new * .5 + overlap * .5 + .001 * proximity)
            state = update(state, options[choice])
            states.append(state)
        elapsed = (perf_counter() - started) * 1000
        # Gold enters only after the complete annotation-free trajectory has been generated.
        path = [(sum(map(len, state)), all(t.gold <= used for t, used in zip(task.trials, state, strict=True)))
                for state in states]
        routes[mode] = dict(scores=scores, path=path, replay_ms=elapsed)
    return routes


def at_round(route, level):
    path = route["path"]
    return path[min(level, len(path)-1)]


def stop(route, cutoff):
    level = 0
    for score in route["scores"]:
        if score < cutoff:
            break
        level += 1
    return at_round(route, level)


def outcomes(paths, policy):
    return np.asarray([policy(route) for route in paths], dtype=float)


def calibrate(paths, target):
    scores = [score for route in paths for score in route["scores"]]
    thresholds = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 201)), max(scores) + 1.])
    eligible = []
    for cutoff in thresholds:
        cost, success = outcomes(paths, lambda route: stop(route, cutoff)).mean(axis=0)
        if cost <= target:
            eligible.append((success, -cost, float(cutoff)))
    successes, negative_cost, cutoff = max(eligible)
    return dict(cutoff=cutoff, dev_complete_rate=float(successes), dev_mean_sentences=float(-negative_cost))


def test(paths, groups, cutoff):
    fixed = [outcomes(paths, lambda route: at_round(route, level)) for level in range(MAX_ROUNDS+1)]
    chosen = outcomes(paths, lambda route: stop(route, cutoff))
    row_cost = float(chosen[:, 0].mean())
    for level, right in enumerate(fixed[1:]):
        if right[:, 0].mean() >= row_cost:
            left = fixed[level]
            lo, hi = left[:, 0].mean(), right[:, 0].mean()
            mixture = (row_cost-lo)/(hi-lo) if hi > lo else 0.
            expected = left[:, 1] * (1-mixture) + right[:, 1] * mixture
            break
    else:
        raise ValueError("Selected route exceeds all fixed depths")
    return dict(
        adaptive=dict(mean_sentences=row_cost, complete=int(chosen[:, 1].sum())),
        fixed=[dict(mean_sentences=float(data[:, 0].mean()), complete=int(data[:, 1].sum())) for data in fixed],
        matched_fixed=dict(rounds=[level, level+1], higher_depth_probability=float(mixture),
                           expected_complete=float(expected.sum()),
                           adaptive_difference=clustered_paired_interval(chosen[:, 1] - expected, groups)),
        exhaustive_policy_replay_p95_ms=float(np.quantile([route["replay_ms"] for route in paths], .95)),
    )


def strongest_matched_fixed(held, groups, mode, cutoff):
    selected = outcomes([row[mode] for row in held], lambda route: stop(route, cutoff))
    budget = float(selected[:, 0].mean())
    fixed = {(method, depth): outcomes([row[method] for row in held],
                                      lambda route: at_round(route, depth))
             for method in CONTROLLERS for depth in range(MAX_ROUNDS + 1)}
    fronts = [(float(values[:, 0].mean()), key, values[:, 1]) for key, values in fixed.items()]
    best = None
    for (lo, low_policy, lower), (hi, high_policy, higher) in combinations(sorted(fronts, key=lambda arm: arm[:2]), 2):
        if not lo <= budget <= hi or hi == lo:
            continue
        mix = (budget - lo) / (hi - lo)
        matched = (1 - mix) * lower + mix * higher
        average = float(matched.mean())
        if best is None or average > best[0]:
            best = (average, low_policy, high_policy, mix, matched)
    if best is None:
        raise ValueError("Cost outside fixed-policy envelope")
    _, low_policy, high_policy, mix, matched = best
    return dict(
        lower_cost_policy=low_policy, higher_cost_policy=high_policy, higher_policy_probability=mix,
        expected_complete=float(matched.sum()),
        paired_difference=clustered_paired_interval(selected[:, 1]-matched, groups),
    )


def run(archive, output):
    if digest(archive) != ARCHIVE or output.exists():
        raise ValueError("Require the pinned archive and a fresh output")
    splits = read_splits(archive)
    model = fit(splits["train"])
    dev = [build_paths(task, model) for task in splits["dev"]]
    test_tasks = splits["Gold_test"]
    held = [build_paths(task, model) for task in test_tasks]
    edges = [dict(case=task.key, doc=trial.paper) for task in test_tasks for trial in task.trials]
    cluster = {edge["case"]: group for edge, group in zip(edges, components(edges), strict=True)}
    groups = [cluster[task.key] for task in test_tasks]
    target = float(np.mean([at_round(row["lexical"], 1)[0] for row in dev]))
    results = {}
    for mode in CONTROLLERS:
        calibrated = calibrate([row[mode] for row in dev], target)
        result = test([row[mode] for row in held], groups, calibrated["cutoff"])
        by_sources = {}
        for count in (1, 2):
            routes = [row[mode] for task, row in zip(test_tasks, held, strict=True) if len(task.trials) == count]
            chosen = outcomes(routes, lambda route: stop(route, calibrated["cutoff"]))
            by_sources[str(count)] = dict(cases=len(chosen), complete=int(chosen[:, 1].sum()),
                                          mean_sentences=float(chosen[:, 0].mean()))
        results[mode] = dict(
            calibration=calibrated, **result, adaptive_by_trial_count=by_sources,
            matched_fixed_envelope=strongest_matched_fixed(held, groups, mode, calibrated["cutoff"]),
        )
    report = dict(
        archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
        cohort={split: len(tasks) for split, tasks in splits.items()},
        unique_trial_counts={split: len({trial.paper for task in tasks for trial in task.trials})
                             for split, tasks in splits.items()},
        dev_target_mean_sentences=target, training_actions=model[2],
        learned_coefficients=model[0].tolist(), learned_intercept=model[1], max_rounds=MAX_ROUNDS,
        results=results,
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
