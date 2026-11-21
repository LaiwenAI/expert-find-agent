"""Transfer reviewer-relevance weighting to matched CoF evidence scheduling."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.cof_deep_source_routing import (
    attachments, initial_reads, logistic, simulate, status_map, train,
)
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, fit

READS = 5
ARMS = ("dense_state", "learned_acceptance", "rank_prior", "observed_relevance", "judged_relevance")


def owner_features(row):
    shallow = initial_reads(row)
    features = np.asarray(row["features"], dtype=float)
    accepted = [sum(bool(row["accepts"][doc]) for doc in docs if doc in shallow)
                for docs in row["candidates"]]
    matrix = []
    for rank, docs in enumerate(row["candidates"], 1):
        visible = [doc for doc in docs if doc in shallow]
        if not visible:
            raise ValueError("Expected at least one shallow document per reviewer")
        summary = features[visible][:, [1, 2, 3, 5]]
        matrix.append([np.log(rank), len(docs), len(visible), accepted[rank - 1],
                       *summary.max(axis=0), *summary.mean(axis=0)])
    return np.asarray(matrix)


def train_reviewers(source, target_titles):
    x, labels = [], []
    for row in source["data"]:
        if row["title"] in target_titles:
            continue
        x.append(owner_features(row))
        labels.extend(-1 if grade is None else int(grade > 0) for grade in row["truth"])
    values, y = np.concatenate(x), np.asarray(labels)
    judged = y >= 0
    return fit(values, y == 1), fit(values[judged], y[judged]), {
        "owner_slots": len(y), "judged_slots": int(judged.sum()), "positive_slots": int((y == 1).sum()),
    }


def source_order(profiles, observed, probabilities, utility):
    counts = status_map(profiles, observed)
    priority = Counter()
    for owner, profile in enumerate(profiles):
        needed = 2 - counts[owner]
        remaining = [doc for doc in profile if doc not in observed]
        if not needed or len(remaining) < needed:
            continue
        if needed == 1:
            for doc in remaining:
                priority[doc] += utility[owner] * probabilities[doc]
            continue
        best, second = sorted(remaining, key=lambda doc: (-probabilities[doc], doc))[:2]
        for doc in remaining:
            partner = second if doc == best else best
            priority[doc] += utility[owner] * probabilities[doc] * probabilities[partner]
    return sorted(priority, key=lambda doc: (-priority[doc], doc))


def route(row, probabilities, utility, read_cap, token_cap):
    initial = initial_reads(row)
    observed = {doc: row["accepts"][doc] for doc in sorted(initial)}
    fallback = sorted(range(len(probabilities)), key=lambda doc: (-probabilities[doc], doc))
    tokens, reads = 0, 0
    while reads < read_cap:
        ordered = source_order(row["candidates"], observed, probabilities, utility)
        doc = next((doc for doc in [*ordered, *fallback] if doc not in observed and
                    tokens + row["token_lengths"][doc] <= token_cap), None)
        if doc is None:
            break
        observed[doc] = row["accepts"][doc]
        tokens += row["token_lengths"][doc]
        reads += 1
    result = attachments(row["candidates"], initial, observed)
    return dict(citations=result, cited=sum(bool(value) for value in result), read_count=reads, tokens=tokens)


def selected_slots(record, cap=None, preserve=()):
    issued = {i for i, docs in enumerate(record["citations"]) if docs}
    if cap is None:
        return issued
    fixed = set(preserve)
    if not fixed <= issued or cap < len(fixed):
        raise ValueError("Cannot preserve the shallow citation package at the matched cap")
    return fixed | set(sorted(issued - fixed)[:cap-len(fixed)])


def shallow_slots(row):
    initial = initial_reads(row)
    observed = {doc: row["accepts"][doc] for doc in initial}
    citations = attachments(row["candidates"], initial, observed)
    return {i for i, docs in enumerate(citations) if docs}


def judged_concordance(scores, truth):
    good = [score for score, grade in zip(scores, truth, strict=True) if grade is not None and grade > 0]
    bad = [score for score, grade in zip(scores, truth, strict=True) if grade is not None and grade <= 0]
    return sum(int(a > b) + .5 * int(a == b) for a in good for b in bad), len(good) * len(bad)


def judgements(slots, truth):
    return Counter(
        cited=len(slots),
        known=sum(truth[i] is not None and truth[i] > 0 for i in slots),
        negative=sum(truth[i] is not None and truth[i] <= 0 for i in slots),
        unknown=sum(truth[i] is None for i in slots),
    )


def evaluate(source, target):
    target_titles = {row["title"] for row in target["data"]}
    accept, source_fit = train(source, target_titles)
    response, judged, relevance_fit = train_reviewers(source, target_titles)
    totals, matched = {arm: Counter() for arm in ARMS}, {arm: Counter() for arm in ARMS}
    contrasts = {arm: [] for arm in ARMS}
    partial = {arm: [0, 0] for arm in ARMS}
    captures = []
    concordance = {name: Counter() for name in ("rank_prior", "observed_relevance", "judged_relevance")}
    order_only = {arm: Counter() for arm in ARMS}
    for row in target["data"]:
        base = simulate(row, "dense_state", accept, READS)
        read_cap, token_cap = base["read_count"], base["tokens"]
        probabilities = logistic(accept[0] + np.asarray(row["features"]) @ accept[1:])
        owner = owner_features(row)
        utilities = {
            "learned_acceptance": np.ones(len(owner)),
            "rank_prior": 1 / np.arange(1, len(owner) + 1),
            "observed_relevance": estimate(response, owner),
            "judged_relevance": estimate(judged, owner),
        }
        for name, counts in concordance.items():
            right, total = judged_concordance(utilities[name], row["truth"])
            counts.update(correct=right, pairs=total)
        policies = {ARMS[0]: {key: base[key] for key in ("citations", "cited", "read_count", "tokens")}}
        for arm, util in utilities.items():
            # A floor keeps low-scoring or unjudged owners eligible for a read.
            weights = np.ones(len(util)) if arm == "learned_acceptance" else .25 + .75 * util / max(util)
            policies[arm] = route(row, probabilities, weights, read_cap, token_cap)
        cap = min(record["cited"] for record in policies.values())
        preserve = shallow_slots(row)
        reference = selected_slots(policies["dense_state"], cap, preserve)
        reference_grade = judgements(reference, row["truth"])
        for arm, record in policies.items():
            if record["read_count"] > read_cap or record["tokens"] > token_cap:
                raise ValueError("Follow-up route exceeded the original workload")
            totals[arm].update(judgements(selected_slots(record), row["truth"]))
            totals[arm].update(reads=record["read_count"], tokens=record["tokens"])
            order_only[arm].update(judgements(selected_slots(record, cap), row["truth"]))
            kept = selected_slots(record, cap, preserve)
            counts = judgements(kept, row["truth"])
            matched[arm].update(counts)
            delta = counts["known"] - reference_grade["known"]
            contrasts[arm].append(delta)
            partial[arm][0] += delta - sum(row["truth"][i] is None for i in reference - kept)
            partial[arm][1] += delta + sum(row["truth"][i] is None for i in kept - reference)
        captures.append(policies)
    summary = {}
    count = len(target["data"])
    for arm in ARMS:
        summary[arm] = dict(
            full=totals[arm], matched_issuance=matched[arm], matched_by_rank_only=order_only[arm],
            paired_matched_p5_delta_pp=paired_interval([20 * value for value in contrasts[arm]]),
            matched_p5_partial_label_interval_pp=[20 * value / count for value in partial[arm]],
        )
    return dict(target=target["dataset"], training=source["dataset"], queries=count,
                acceptance_fit=source_fit, relevance_fit=relevance_fit, outcomes=summary,
                within_query_judged_concordance=concordance,
                signature=sha256(json.dumps(captures, sort_keys=True).encode()).hexdigest())


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm compute allocation and a fresh output")
    expected = json.loads(args.reference.read_bytes())
    inputs = [args.kdd, args.scirepeval]
    panels = [json.loads(path.read_bytes()) for path in inputs]
    provided = {panel["dataset"]: digest(path) for panel, path in zip(panels, inputs, strict=True)}
    if provided != expected["capture_sha256"]:
        raise ValueError("Inputs must match the archived CoF captures")
    kdd, sci = panels
    pairs = [(sci, kdd), (kdd, sci)]
    if args.reverse:
        pairs.reverse()
    results = {target["dataset"]: evaluate(source, target) for source, target in pairs}
    write_json(args.output, dict(results=results, inputs=provided, reverse=args.reverse,
                                 runner=digest(Path(__file__)), slurm=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("kdd", "scirepeval", "reference", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
