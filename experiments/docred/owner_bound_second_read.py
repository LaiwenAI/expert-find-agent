"""Test whether naming the candidate improves budget-matched gap recovery."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import mixed_role_window_coverage as panels
from experiments.followir.core17_reranking import digest, write_json


LEARNER = panels.learner
BASELINE = "pooled_gap_pooled_search"
ARMS = (
    BASELINE, "pooled_gap_owner_search", "owner_gap_pooled_search",
    "owner_gap_owner_search", "parallel_three_sentences",
)
COVERAGES = (.5, 1.)


def add_coverage(proof, candidate):
    return all(proof[i] for i in panels.BASE.PAIR[candidate])


def atomic_read(case, read, groups):
    prompts = []
    for claim, window in groups:
        work, person = case["names"][claim]
        role = panels.NAMES.ROLES[LEARNER.role_at(case, claim)]
        question = f"Does the source explicitly say that {person} had the role of {role} on the work {work}?"
        text = "\n".join(case["sentences"][i] for i in sorted(window))
        prompts.append((question, text))
    values, clipped = read(prompts)
    if clipped or not np.isfinite(values).all():
        raise ValueError("Invalid reader result")
    return np.asarray(values)


def initial_anchors(group, scoped):
    first = int(group.sum(axis=1).argmax())
    utility = np.maximum(group, group[first]).sum(axis=1)
    utility[first] = -np.inf
    second = int(utility.argmax())
    two = np.array(sorted((first, second)))
    chosen = scoped[two].argmax(axis=0)
    if np.all(chosen == chosen[0]):
        other = 1 - chosen[0]
        flip = np.argmax(scoped[two[other]] - scoped[two[chosen[0]]])
        chosen[flip] = other
    return two, two[chosen]


def annotation_complete(case, windows):
    return [
        bool(gold) and set(gold) <= set(window)
        for gold, window in zip(case["annotated"], windows, strict=True)
    ]


def decision(case, margins, proof, elapsed):
    worst = np.minimum(margins[:2], margins[2:])
    tie = int(panels.NAMES.stable(case["id"], "candidate")[-1], 16) % 2
    best = tie if worst[0] == worst[1] else int(worst[1] > worst[0])
    return dict(
        confidence=(float(worst[best]), panels.NAMES.stable(case["id"], "coverage")),
        top=best, known=case["known"][best], top_proof=add_coverage(proof, best),
        proofs=sum(add_coverage(proof, i) for i in range(2)),
        elapsed_ms=elapsed,
    )


def recovery(case, read, initial, initial_ms, anchors, unseen, scores, arm):
    start = perf_counter()
    leading = decision(case, initial, (False,) * 4, 0)["top"]
    strength = initial.reshape(2, 2)
    scope = strength[:, leading] if arm.startswith("owner_gap") else strength.max(axis=1)
    condition = int(scope.argmin())
    claims = np.array([2 * condition, 2 * condition + 1])
    retrieval_scores = scores[:, claims[leading]] if arm.endswith("owner_search") else scores[:, claims].max(axis=1)
    added = int(unseen[retrieval_scores.argmax()])
    windows = [(int(anchor), added) if claim in claims else (int(anchor),)
               for claim, anchor in enumerate(anchors)]
    next_margins = initial.copy()
    next_margins[claims] = atomic_read(case, read, [(i, windows[i]) for i in claims])
    elapsed = initial_ms + (perf_counter() - start) * 1000
    proof = annotation_complete(case, windows)
    row = decision(case, next_margins, proof, elapsed)
    return dict(row, condition=condition, added=added)


def parallel(case, read):
    start = perf_counter()
    group, scoped = LEARNER.lexical_features(case)
    two, anchors = initial_anchors(group, scoped)
    utility = np.maximum(group, group[two].max(axis=0)).sum(axis=1)
    utility[two] = -np.inf
    third = int(utility.argmax())
    other = [int(two[0]) if anchor == two[1] else int(two[1]) for anchor in anchors]
    alternatives = [(int(anchor), third if scoped[third, i] >= scoped[other[i], i] else other[i])
                    for i, anchor in enumerate(anchors)]
    if all(third not in pair for pair in alternatives):
        i = int(np.argmax(scoped[third] - scoped[other, np.arange(4)]))
        alternatives[i] = (int(anchors[i]), third)
    if len(set().union(*map(set, alternatives))) != 3:
        raise AssertionError("Parallel route changed the unique sentence budget")
    scores = atomic_read(case, read, list(enumerate(alternatives)))
    return decision(case, scores, annotation_complete(case, alternatives),
                    (perf_counter() - start) * 1000)


def collect(rows, weights, read, offset):
    outcomes = []
    for index, case in enumerate(rows):
        started = perf_counter()
        group, scoped = LEARNER.lexical_features(case)
        two, anchors = initial_anchors(group, scoped)
        first_windows = [(int(anchor),) for anchor in anchors]
        initial = atomic_read(case, read, list(enumerate(first_windows)))
        initial_ms = (perf_counter() - started) * 1000
        first = decision(case, initial, annotation_complete(case, first_windows), initial_ms)
        started = perf_counter()
        left, right, feats = LEARNER.pair_features(case)
        combinations = np.einsum("ifc,f->ic", feats, weights)
        size = len(group)
        pairs = np.full((size, size), -1)
        pairs[left, right] = pairs[right, left] = np.arange(len(left))
        unseen = np.setdiff1d(np.arange(size), two)
        scores = combinations[pairs[unseen[:, None], anchors], np.arange(4)]
        retrieval_setup_ms = (perf_counter() - started) * 1000
        rotation = (index + offset) % len(ARMS)
        result = {}
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            result[arm] = (parallel(case, read) if arm == "parallel_three_sentences"
                           else recovery(case, read, initial, initial_ms + retrieval_setup_ms,
                                         anchors, unseen, scores, arm))
        # Diagnose the proof reachable by any permitted second-stage action.
        reachable = [first["proofs"]]
        for condition in range(2):
            for sentence in unseen:
                windows = [(*window, int(sentence)) if i // 2 == condition else window
                           for i, window in enumerate(first_windows)]
                proof = annotation_complete(case, windows)
                reachable.append(sum(add_coverage(proof, i) for i in range(2)))
        outcomes.append((case["doc"], first, result, max(reachable)))
    return outcomes


def summarize(outcomes, calibration):
    titles = [title for title, *_ in outcomes]
    groups = {arm: [results[arm] for _, _, results, _ in outcomes] for arm in ARMS}
    diagnostics = {}
    totals = {}
    for arm, rows in groups.items():
        counts = Counter()
        for (_, first, other, _), row in zip(outcomes, rows, strict=True):
            counts.update(
                known_gains=int(row["known"] and not first["known"]),
                known_losses=int(first["known"] and not row["known"]),
                top_changed=int(first["top"] != row["top"]),
                completed_proofs=int(row["proofs"] - first["proofs"]),
            )
            if "added" in row:
                reference = other[BASELINE]
                counts.update(action_changed=int(
                    (row["condition"], row["added"]) != (reference["condition"], reference["added"])
                ))
        diagnostics[arm] = dict(counts)
        totals[arm] = dict(p95_ms=float(np.quantile([row["elapsed_ms"] for row in rows], .95)))
    for label in ("matched_issuance", "locked_development"):
        curves = {}
        for fraction in COVERAGES:
            cells, known = {}, {}
            for arm, values in groups.items():
                comparison = values if label == "matched_issuance" else [rows[arm] for _, _, rows, _ in calibration]
                floor = max(1, round(len(comparison) * fraction))
                threshold = sorted((row["confidence"] for row in comparison), reverse=True)[floor - 1]
                accepted = [fraction == 1 or row["confidence"] >= threshold for row in values]
                known[arm] = np.asarray([int(a and row["known"]) for a, row in zip(accepted, values, strict=True)])
                cells[arm] = dict(issued=sum(accepted), known=int(known[arm].sum()),
                                  top_proof=sum(a and row["top_proof"] for a, row in zip(accepted, values, strict=True)),
                                  complete_proofs=int(sum(row["proofs"] for row in values)))
            for arm in cells:
                cells[arm]["paired_known_delta_vs_pooled"] = clustered_paired_interval(
                    known[arm] - known[BASELINE], titles,
                )
            curves[str(fraction)] = cells
        totals[label] = curves
    return dict(
        cases=len(outcomes), titles=len(set(titles)), runtime=totals,
        initial_known=sum(row["known"] for _, row, _, _ in outcomes),
        initial_proofs=sum(row["proofs"] for _, row, _, _ in outcomes),
        reachable_proofs=sum(oracle for _, _, _, oracle in outcomes),
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for args in ("release", "checkpoint", "output"):
        parser.add_argument(args, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm allocation")
    same, mixed, hashes, weights, training = panels.load(args.release)
    tokenizer, model, provenance = panels.BASE.reader.load_reader(args.checkpoint)
    read = panels.BASE.reader.reader_scorer(tokenizer, model, system=panels.BASE.shapes.SYSTEM)
    collect(same["calibration"][:1], weights, read, 0)
    results = {}
    for family, split in (("same", same), ("mixed", mixed)):
        calibration = collect(split["calibration"], weights, read, args.order_offset)
        for part in ("test", "external"):
            observations = collect(split[part], weights, read, args.order_offset)
            results[f"{family}_{part}"] = summarize(observations, calibration)
            print(f"Completed {family} {part}: {len(observations)}", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], inputs=hashes, training=training,
        weights=weights.tolist(), checkpoint=provenance["revision"], order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), groups=results,
    ))


if __name__ == "__main__":
    main()
