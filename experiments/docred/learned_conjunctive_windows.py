"""Learn evidence-pair retrieval under a shared four-sentence budget."""

import argparse
from collections import Counter
from itertools import combinations, product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import conjunctive_proof_windows as base
from experiments.docred import person_bound_evidence as names
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("global_names", "lexical_pair", "learned_sum", "learned_conjunction")
LOCAL_PAIRS = np.array(list(combinations(range(4), 2)))
ASSIGNMENTS = np.array([
    assignment for assignment in product(range(6), repeat=4)
    if len(set(LOCAL_PAIRS[list(assignment)].flat)) == 4
])
CLAIMS = np.arange(4)


def pair_features(case):
    words, _, idf = names.profile(case["sentences"])
    left, right = np.triu_indices(len(words), 1)
    role = names.tokens(names.ROLE_CUES[case["role"]])
    cue = np.array([names.coverage(role, row, idf) for row in words])
    entities = {}
    for name in dict.fromkeys(name for pair in case["names"] for name in pair):
        tokens = tuple(names.tokens(name))
        entities[name] = np.array([
            (float(names.contains(row, tokens)), names.coverage(tokens, row, idf))
            for row in words
        ])
    features = []
    for i, (work, owner) in enumerate(case["names"]):
        other_owner = case["names"][i ^ 1][1]
        other_work = case["names"][i ^ 2][0]
        a, b = entities[work], entities[owner]
        rival = entities[other_owner][:, 0]
        alternative = entities[other_work][:, 0]
        local = a[:, 1] * b[:, 1]
        features.append(np.column_stack((
            np.maximum(a[left], a[right]), np.maximum(b[left], b[right]),
            np.minimum(a[left], a[right]), np.minimum(b[left], b[right]),
            np.maximum(local[left], local[right]),
            np.maximum(cue[left], cue[right]),
            np.maximum(cue[left] * b[left, 1], cue[right] * b[right, 1]),
            np.maximum(cue[left] * a[left, 1], cue[right] * a[right, 1]),
            np.maximum(rival[left], rival[right]), np.maximum(alternative[left], alternative[right]),
            1 / (right - left), 1 / (1 + left), 1 / (1 + right),
        )))
    return left, right, np.stack(features, axis=-1)


def fit(release, panels):
    blocked = {row["doc"] for rows in panels.values() for row in rows}
    train = json.loads((release / "train_revised.json").read_bytes())
    for split in ("dev", "test"):
        blocked.update(row["title"] for row in json.loads((release / f"{split}_revised.json").read_bytes()))
    normal = None
    target = None
    used = set()
    examples = 0
    for case in base.cases(row for row in train if row["title"] not in blocked):
        left, right, inputs = pair_features(case)
        _, scope = base.features(case)
        lexical = np.maximum(scope[left], scope[right])
        for claim, gold in enumerate(case["annotated"]):
            if not 0 < len(gold) <= 2:
                continue
            positives = (np.isin(left, gold).astype(int) + np.isin(right, gold)) == len(gold)
            if not np.any(positives) or np.all(positives):
                continue
            order = np.argsort(-lexical[:, claim], kind="stable")
            good = order[positives[order]][:4]
            alternatives = order[~positives[order]][:12]
            difference = inputs[good, :, claim][:, None, :] - inputs[alternatives, :, claim][None, :, :]
            matrix = difference.reshape(-1, inputs.shape[1])
            if normal is None:
                normal = np.eye(matrix.shape[1])
                target = np.zeros(matrix.shape[1])
            normal += matrix.T @ matrix / len(matrix)
            target += matrix.mean(axis=0)
            used.add(case["doc"])
            examples += 1
    if not examples:
        raise ValueError("No disjoint training annotations")
    return np.linalg.solve(normal, target), dict(documents=len(used), annotated_claims=examples)


def choose(scores, left, right, reference, conjunction):
    # Each hypothesis starts from the union of two high-scoring witness pairs.
    pool = {int(i) for col in scores.T for i in np.argsort(-col, kind="stable")[:4]}
    proposals = {tuple(reference)}
    for i, j in combinations(sorted(pool), 2):
        evidence = set((int(left[i]), int(right[i]), int(left[j]), int(right[j])))
        for sentence in reference:
            if len(evidence) == base.WIDTH:
                break
            evidence.add(sentence)
        proposals.add(tuple(sorted(evidence)))
    sets = np.array(sorted(proposals))
    n = int(right.max() + 1)
    pair_index = np.full((n, n), -1)
    pair_index[left, right] = np.arange(len(left))
    pairs = pair_index[sets[:, LOCAL_PAIRS[:, 0]], sets[:, LOCAL_PAIRS[:, 1]]]
    values = scores[pairs][:, ASSIGNMENTS, CLAIMS]
    total = values.sum(axis=-1)
    if conjunction:
        primary = base.weakest(values)
        best = np.where(primary == primary.max(), total, -np.inf).argmax()
    else:
        best = total.argmax()
    si, ai = np.unravel_index(best, total.shape)
    return [tuple(row) for row in sets[si][LOCAL_PAIRS[ASSIGNMENTS[ai]]]]


def candidate_windows(case, weights, arm):
    group, scope = base.features(case)
    reference = base.choose_windows(group, scope, "global_names")
    if arm == "global_names":
        return base.assign_windows(scope, reference)
    if arm == "lexical_pair":
        left, right = np.triu_indices(len(scope), 1)
        scores = np.maximum(scope[left], scope[right])
    else:
        left, right, values = pair_features(case)
        scores = np.einsum("ifc,f->ic", values, weights)
    return choose(scores, left, right, reference, arm == "learned_conjunction")


def collect(panel, weights, read, offset):
    outcomes = []
    for i, case in enumerate(panel):
        rows = {}
        shift = (offset + i) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            selected = candidate_windows(case, weights, arm)
            selection_ms = (perf_counter() - start) * 1000
            if len(set().union(*map(set, selected))) != 4 or any(len(set(pair)) != 2 for pair in selected):
                raise AssertionError("A policy changed the evidence budget")
            snippets = ["\n".join(case["sentences"][j] for j in pair) for pair in selected]
            available = [
                bool(gold) and set(gold) <= set(pair)
                for gold, pair in zip(case["annotated"], selected, strict=True)
            ]
            row = dict(available=available, selection_ms=selection_ms)
            if read is not None:
                start = perf_counter()
                scores, clipped = read([
                    (f"Does the source explicitly say that {person} had the role of {names.ROLES[case['role']]} "
                     f"on the work {work}?", text)
                    for (work, person), text in zip(case["names"], snippets, strict=True)
                ])
                if clipped or not np.isfinite(scores).all():
                    raise ValueError("Invalid reader result")
                row.update(margins=tuple(min(scores[j] for j in pair) for pair in base.PAIR),
                           reader_ms=(perf_counter() - start) * 1000)
            rows[arm] = row
        outcomes.append((case, rows))
    return outcomes


def summarize(outcomes, thresholds):
    metrics, proof, known_top = {}, {}, {}
    clusters = [case["doc"] for case, _ in outcomes]
    for arm in ARMS:
        counts = Counter()
        full, fixed = [], []
        for case, rows in outcomes:
            row = rows[arm]
            complete = [all(row["available"][i] for i in pair) for pair in base.PAIR]
            full.append(sum(complete))
            counts.update(
                known_candidates=sum(case["known"]),
                atomic_annotated=sum(bool(gold) for gold in case["annotated"]),
                atomic_proofs=sum(row["available"]),
                both_annotations=sum(all(case["annotated"][i] for i in pair) for pair in base.PAIR),
                conjunction_proofs=sum(complete),
            )
            if thresholds is not None:
                margin = row["margins"]
                top = int(margin[1] > margin[0])
                accepted = [m > thresholds[arm] for m in margin]
                fixed.append(int(case["known"][top]))
                counts.update(top_one_known=fixed[-1], top_one_proof=int(complete[top]),
                              accepted_known=sum(a and k for a, k in zip(accepted, case["known"], strict=True)),
                              accepted_unjudged=sum(a and not k for a, k in zip(accepted, case["known"], strict=True)),
                              accepted_with_proof=sum(a and c for a, c in zip(accepted, complete, strict=True)))
        metrics[arm] = dict(counts, selector_p95_ms=float(np.quantile(
            [rows[arm]["selection_ms"] for _, rows in outcomes], .95
        )))
        proof[arm] = np.array(full) / 2
        if thresholds is not None:
            known_top[arm] = np.array(fixed)
            metrics[arm].update(cutoff=thresholds[arm], combined_p95_ms=float(np.quantile(
                [rows[arm]["reader_ms"] + rows[arm]["selection_ms"] for _, rows in outcomes], .95
            )))
    for arm in ARMS:
        metrics[arm]["paired_proof_delta_vs_global"] = clustered_paired_interval(
            proof[arm] - proof["global_names"], clusters,
        )
        if thresholds is not None:
            metrics[arm]["paired_top_one_delta_vs_global"] = clustered_paired_interval(
                known_top[arm] - known_top["global_names"], clusters,
            )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm allocation")
    panels, hashes = base.load(args.release)
    weights, training = fit(args.release, panels)
    read, revision = None, None
    if args.checkpoint:
        tokenizer, model, revision = base.reader.load_reader(args.checkpoint)
        read = base.reader.reader_scorer(tokenizer, model, system=base.shapes.SYSTEM)
        collect(panels["calibration"][:1], weights, read, 0)
    results = {label: collect(panel, weights, read, args.order_offset) for label, panel in panels.items()}
    thresholds = {arm: base.cutoff(results["calibration"], arm) for arm in ARMS} if read else None
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], inputs=hashes, training=training,
        weights=weights.tolist(), checkpoint=revision["revision"] if revision else None, order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)),
        groups={label: summarize(rows, thresholds) for label, rows in results.items()},
    ))


if __name__ == "__main__":
    main()
