"""Evaluate selective identity linkage with public gold author clusters."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from time import perf_counter_ns
from typing import NamedTuple

import numpy as np

from prepare import key


RISK = 0.05
MIN_ISSUED = 50
METHODS = ("name_unique", "affiliation_unique", "coauthor_unique", "both_unique", "rank_all", "cascade", "coauthor_cascade")
THRESHOLDS = np.r_[np.arange(-0.25, 1.51, .025), np.inf]


class Mention(NamedTuple):
    paper: str
    year: int
    name: str
    first: str
    middle: str
    affiliations: frozenset
    coauthors: frozenset
    title: frozenset
    venue: str

    @classmethod
    def parse(cls, record, missing):
        return cls(
            record["paper"], record["year"], record["name"], record["first"], record["middle"],
            frozenset() if missing == "affiliation" else frozenset(record["affiliations"]),
            frozenset() if missing == "coauthors" else frozenset(record["coauthors"]),
            frozenset(record["title"]), record["venue"],
        )


class Block:
    def __init__(self, keys, *, missing):
        self.profiles = [tuple(Mention.parse(record, missing) for record in owner["profile"]) for owner in keys]
        self.queries = [(Mention.parse(record, missing), i) for i, owner in enumerate(keys)
                        for record in owner["queries"]]
        self.by_affiliation = defaultdict(set)
        self.by_coauthor = defaultdict(set)
        self.by_name = defaultdict(set)
        self.by_paper = defaultdict(set)
        for i, profile in enumerate(self.profiles):
            for record in profile:
                self.by_paper[record.paper].add(i)
                for affiliation in record.affiliations:
                    self.by_affiliation[affiliation].add(i)
                for author in record.coauthors:
                    self.by_coauthor[author].add(i)
                if record.name:
                    self.by_name[record.name].add(i)

    def exact(self, query):
        affiliation = set().union(*(self.by_affiliation.get(x, ()) for x in query.affiliations))
        coauthor = set().union(*(self.by_coauthor.get(x, ()) for x in query.coauthors))
        present = self.by_paper.get(query.paper, set())
        affiliation -= present
        coauthor -= present
        name = self.by_name.get(query.name, set()) - present
        return name, affiliation, coauthor, affiliation & coauthor & name


def jaccard(left, right):
    return len(left & right) / len(left | right) if left and right else 0.0


def features(query, reference):
    same_affiliation = bool(query.affiliations & reference.affiliations)
    same_coauthor = len(query.coauthors & reference.coauthors)
    same_name = bool(query.name and query.name == reference.name)
    return [
        1.0, same_name, bool(query.first and query.first == reference.first),
        bool(query.middle and reference.middle and query.middle != reference.middle),
        same_affiliation, bool(same_coauthor), min(same_coauthor, 5) / 5,
        jaccard(query.coauthors, reference.coauthors),
        jaccard(query.title, reference.title), bool(query.venue and query.venue == reference.venue),
        1 / (1 + abs(query.year - reference.year)) if query.year and reference.year else 0.,
        same_affiliation and bool(same_coauthor),
    ]


def fit(blocks):
    x, y = [], []
    for block in blocks.values():
        for query, target in block.queries:
            present = block.by_paper.get(query.paper, set())
            for i, profile in enumerate(block.profiles):
                if i in present:
                    continue
                x.extend(features(query, anchor) for anchor in profile)
                y.extend(int(target == i) for _ in profile)
    matrix = np.asarray(x)
    labels = np.asarray(y)
    weights = np.where(labels == 1, np.sqrt(len(labels) / max(1, labels.sum())), 1.0)
    weighted = matrix.T * weights
    penalty = np.eye(matrix.shape[1]) * 5
    penalty[0, 0] = 0.0
    return np.linalg.solve(weighted @ matrix + penalty, weighted @ labels)


def unique(candidates):
    return next(iter(candidates)) if len(candidates) == 1 else -1


def ranker(block, query, model):
    ids, vectors = [], []
    present = block.by_paper.get(query.paper, set())
    for i, profile in enumerate(block.profiles):
        if i in present:
            continue
        for reference in profile:
            ids.append(i)
            vectors.append(features(query, reference))
    if not ids:
        return -1, -np.inf, 0
    scores = defaultdict(lambda: -np.inf)
    for candidate, score in zip(ids, np.asarray(vectors) @ model, strict=True):
        scores[candidate] = max(scores[candidate], score)
    top = sorted(scores, key=lambda x: (-scores[x], x))
    selected = top[0]
    gap = scores[selected] - scores[top[1]] if len(top)>1 else scores[selected]
    return selected, min(scores[selected], gap), len(ids)


def predict(blocks, model):
    rows = []
    for name, block in sorted(blocks.items()):
        for query, target in block.queries:
            start = perf_counter_ns()
            exact = block.exact(query)
            unique_matches = [unique(option) for option in exact]
            exact_time = perf_counter_ns() - start
            start = perf_counter_ns()
            scored, certainty, operations = ranker(block, query, model)
            rank_time = perf_counter_ns() - start
            predictions = {
                arm: (found, np.inf, 0, exact_time)
                for arm, found in zip(METHODS[:4], unique_matches, strict=True)
            }
            predictions["rank_all"] = scored, certainty, operations, rank_time
            for arm, fast in (("cascade", unique_matches[3]), ("coauthor_cascade", unique_matches[2])):
                predictions[arm] = (fast, np.inf, 0, exact_time) if fast >= 0 else (
                    scored, certainty, operations, rank_time + exact_time,
                )
            rows.append(dict(block=name, known_target=bool(block.profiles[target]), target=target,
                             predictions=predictions))
    return rows


def wilson_upper(wrong, issued):
    if not issued:
        return 1.0
    z = 1.6448536269514722
    point = wrong / issued
    return (point + z*z/(2*issued) + z*sqrt(point*(1-point)/issued + z*z/(4*issued*issued)))/(1+z*z/issued)


def select(rows, method):
    candidates = [(row["predictions"][method][1], row["predictions"][method][0] != row["target"])
                  for row in rows if row["predictions"][method][0] >= 0]
    if not candidates:
        return None
    certainty, wrong = map(np.asarray, zip(*candidates, strict=True))
    for threshold in THRESHOLDS:
        accepted = certainty >= threshold
        issued = int(accepted.sum())
        if issued >= MIN_ISSUED and wilson_upper(int(wrong[accepted].sum()), issued) <= RISK:
            return float(threshold)
    return None


def grouped_interval(counts, seed=20261121):
    rng = np.random.default_rng(seed)
    values = np.asarray(list(counts.values()), dtype=int)
    if not len(values) or not values[:, 0].sum():
        return None
    draws = rng.integers(len(values), size=(2000, len(values)))
    sums = values[draws].sum(axis=1)
    valid = sums[:,0]>0
    return list(map(float, np.quantile(sums[valid,1]/sums[valid,0], [.025,.975])))


def score(rows, method, threshold):
    counts = Counter(queries=len(rows), profiles_available=sum(row["known_target"] for row in rows))
    by_block = defaultdict(lambda: [0, 0])
    times = []
    for row in rows:
        target, certainty, work, time = row["predictions"][method]
        counts["pair_evaluations"] += work
        counts["direct_path"] += method.endswith("cascade") and work == 0 and target >= 0
        times.append(time / 1e6)
        cell = by_block[row["block"]]
        if target >= 0 and threshold is not None and certainty >= threshold:
            wrong = target != row["target"]
            counts["issued"] += 1
            counts["wrong"] += wrong
            counts["correct"] += not wrong
            counts["wrong_missing_profile"] += wrong and not row["known_target"]
            cell[0] += 1
            cell[1] += wrong
    issued = counts["issued"]
    return dict(counts, blocks=len(by_block), risk=counts["wrong"]/issued if issued else None,
                wilson_upper=wilson_upper(counts["wrong"], issued) if issued else None,
                block_bootstrap_interval=grouped_interval(by_block),
                p95_selection_ms=float(np.quantile(times,.95)) if times else None)


def paper_ids(blocks):
    return {
        str(record["paper"])
        for owners in blocks.values() for owner in owners for name in ("profile", "queries") for record in owner[name]
    }


def make(blocks, *, missing=None):
    return {name: Block(block, missing=missing) for name, block in blocks.items()}


def evaluate(data):
    results = {}
    for source, entry in data.items():
        fit_raw = {name: block for name, block in entry["blocks"].items() if key(name)%5 in (0,1,2)}
        calibration_raw = {name: block for name, block in entry["blocks"].items() if key(name)%5 == 3}
        test_raw = {name: block for name, block in entry["blocks"].items() if key(name)%5 == 4}
        model = fit(make(fit_raw))
        calibration = predict(make(calibration_raw), model)
        gates = {method: select(calibration, method) for method in METHODS}
        consumed = fit_raw | calibration_raw
        seen_papers = paper_ids(consumed)
        tests = {}
        for target, target_entry in data.items():
            if target == source:
                target_raw = test_raw
            else:
                target_raw = {name: block for name, block in target_entry["blocks"].items()
                              if name not in consumed and not seen_papers & paper_ids({name:block})}
            tests[target] = {}
            for mode, missing in (("native", None), ("without_affiliation", "affiliation"),
                                  ("without_coauthors", "coauthors")):
                held = predict(make(target_raw, missing=missing), model)
                tests[target][mode] = {
                    "fixed": {method: score(held, method, -np.inf) for method in METHODS[:4]},
                    "thresholded": {method: score(held, method, gates[method]) for method in METHODS},
                }
        results[source] = dict(fit_blocks=len(fit_raw), calibration_blocks=len(calibration_raw),
                               gates={k: "infinity" if v == np.inf else v for k,v in gates.items()},
                               calibration={arm: score(calibration, arm, threshold)
                                            for arm, threshold in gates.items()}, tests=tests)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = args.panel.read_bytes()
    data = json.loads(payload)
    report = dict(panel_sha256=sha256(payload).hexdigest(),
                  datasets={name: {k:v for k,v in entry.items() if k != "blocks"} for name, entry in data.items()},
                  results=evaluate(data))
    args.output.write_text(json.dumps(report, separators=(",",":"), allow_nan=False))


if __name__ == "__main__":
    main()
