"""Evaluate criterion-contrast priorities under the same per-topic call budget."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.followir.evaluate_same_passage_routing import cutoff, decisions, margin_features, summarize
from experiments.followir.pooled_verifier_budget import comparison, key


def features(rows, threshold, include_contrast):
    baseline = margin_features(rows, threshold)
    if not include_contrast:
        return baseline
    topic = np.asarray([row["topic_only"] for row in rows])
    revised = np.asarray([row["cheap"] for row in rows])
    drop = topic - revised
    return np.column_stack((baseline, topic, drop, np.abs(drop), drop * (revised > threshold)))


def learned_priority(development, held, thresholds, include_contrast):
    train = features(development, thresholds[0], include_contrast)
    average = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    train = (train - average) / scale
    target = decisions(development, thresholds)[2]
    coeff = np.linalg.solve(train.T @ train + 16 * np.eye(train.shape[1]),
                            train.T @ (target - target.mean()))
    return ((features(held, thresholds[0], include_contrast) - average) / scale) @ coeff


def mixed_recovery(indices, distance, topic, marks, quota, *, shortlist):
    """Reserve one review for a high topic score among remaining items."""
    ranked = sorted(indices, key=lambda i: (distance[i], marks[i]))
    if quota == 0:
        return [], 0
    boundary = ranked[:quota - 1]
    pool = ranked[quota - 1:quota + shortlist - 1] if shortlist else ranked[quota - 1:]
    recovered = max(pool, key=topic.__getitem__)
    return [*boundary, recovered], len(pool)


def evaluate(development, held, thresholds):
    before, after, _ = decisions(held, thresholds)
    revised = np.asarray([row["cheap"] for row in held])
    topic = np.asarray([row["topic_only"] for row in held])
    distance = np.abs(revised - thresholds[0])
    dropout = topic - revised
    marks = [int.from_bytes(key(row), "big") for row in held]
    by_topic = defaultdict(list)
    for i, row in enumerate(held):
        by_topic[row["topic"]].append(i)
    priorities = {
        "boundary": list(zip(-distance, strict=True)),
        "topic_only": list(zip(topic, -distance, strict=True)),
        "contrast_magnitude": list(zip(abs(dropout), -distance, strict=True)),
        "condition_reject_first": list(zip(~before, dropout, -distance, strict=True)),
        "ridge_baseline": list(zip(learned_priority(development, held, thresholds, False), strict=True)),
        "ridge_with_contrast": list(zip(learned_priority(development, held, thresholds, True), strict=True)),
    }
    budget = {}
    for denominator in (4, 2):
        routes = {}
        selected_by = {}
        for name, rank in priorities.items():
            selected = [i for indices in by_topic.values()
                        for i in sorted(indices, key=lambda i: (rank[i], -marks[i]), reverse=True)
                        [:len(indices) // denominator]]
            accepted = before.copy()
            accepted[selected] = after[selected]
            selected_by[name] = selected
            routes[name] = summarize(held, accepted, before, selected, by_topic)
        for name, shortlist in (("mixed_all_remaining", 0), ("mixed_next_four", 4)):
            selected = []
            additional = 0
            for indices in by_topic.values():
                chosen, checks = mixed_recovery(indices, distance, topic, marks,
                                               len(indices) // denominator, shortlist=shortlist)
                selected.extend(chosen)
                additional += checks
            accepted = before.copy()
            accepted[selected] = after[selected]
            selected_by[name] = selected
            routes[name] = dict(summarize(held, accepted, before, selected, by_topic),
                                extra_small_scores=additional)
        boundary = set(selected_by["boundary"])
        missed = {i for i, row in enumerate(held) if row["label"] == "stable_positive"
                  and after[i] and not before[i] and i not in boundary}
        boundary_acceptance = before.copy()
        boundary_indices = selected_by["boundary"]
        boundary_acceptance[boundary_indices] = after[boundary_indices]
        for name, result in routes.items():
            selected = selected_by[name]
            accepted = before.copy()
            accepted[selected] = after[selected]
            result["missed_boundary_positives_reviewed"] = len(missed.intersection(selected))
            paired = comparison(held, accepted, boundary_acceptance, by_topic)
            result["paired_against_boundary"] = {
                outcome: paired[f"{outcome}_document_delta_per_topic"]
                for outcome in ("positive", "invalidated")
            }
        budget[f"1/{denominator}"] = routes
    return budget


def run(base_path, first_path, second_path, reference, output):
    source, first, second, prior = [json.loads(p.read_bytes())
                                    for p in (base_path, first_path, second_path, reference)]
    base_digest = sha256(base_path.read_bytes()).hexdigest()
    if (first["offset"], second["offset"]) != (0, 1) or any(
            first[key] != second[key] for key in first if key != "offset"):
        raise ValueError("GPU topic-only margins did not reproduce")
    if first["prior_sha256"] != base_digest or first["checkpoint"] != source["compact_revision"]:
        raise ValueError("Contrast margins do not match the original scores")
    if source["source_sha256"] != prior["source_sha256"] or source["code_sha256"] != prior["score_script_sha256"]:
        raise ValueError("Changed prior aggregate")
    panels = {name: [dict(row, topic_only=margin) for row, margin in
                     zip(split["rows"], first["margins"][name], strict=True)]
              for name, split in source["panels"].items()}
    development = {k: panels[k] for k in ("core", "news")}
    thresholds = tuple(cutoff(development, name) for name in ("cheap", "costly"))
    record = prior["result"]["transferred_risk_cutoffs"]
    if list(thresholds) != record["cutoffs"]:
        raise ValueError("Changed verify cutoffs")
    train = development["core"] + development["news"]
    held = panels["robust"]
    result = evaluate(train, held, thresholds)
    for budget, routes in result.items():
        for old, new in (("boundary", "boundary"), ("learned", "ridge_baseline")):
            expected = record["budgets"][budget][old]
            if any(routes[new][k] != val for k, val in expected.items()):
                raise ValueError("Historical routing baseline did not reproduce")
    with output.open("x") as stream:
        json.dump(dict(budgets=result, cutoffs=thresholds, rows=len(held), development_rows=len(train),
                       source_sha256=source["source_sha256"], checkpoints=[first["checkpoint"], source["strong_revision"]],
                       original_score_sha256=base_digest, topic_only_script_sha256=first["code_sha256"],
                       analysis_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                       repeated_offsets=[first["offset"], second["offset"]]), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_path", "first_path", "second_path", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
