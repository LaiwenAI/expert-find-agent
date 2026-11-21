"""Choose source reads by estimated yield over distinct experts."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.trec_enterprise.nomination_read_budget import DEPTH, PAIR_CAP, allocate, oracle_ceiling
from experiments.trec_enterprise.nomination_waterfall import PERSON_CAP, expert_votes, lexical_vectors, nominations

ARMS = ("reuse_reference", "fitted_greedy", "fitted_lookahead")
LOOKAHEAD_WIDTH = 24


def features(index, topic, people, owned):
    pairs, values = [], []
    q = index.positions[topic]
    historical = ~(1 << q)
    similarities = np.maximum(index.similarity[q], 0)
    for i, owner in enumerate(people, 1):
        for j, (doc, vote) in enumerate(owned[owner][:DEPTH], 1):
            catalog = index.support[doc]
            link = catalog[owner] & historical
            neighbors = [n for n in range(len(index.topics)) if link & (1 << n)]
            if not neighbors:
                raise ValueError("Held-out source leaked into nomination")
            features_row = (
                math.log(i), math.log(j), math.log1p(len(owned[owner])),
                math.log1p(sum(bool(mask & historical) for mask in catalog.values())),
                math.log1p(len(neighbors)), vote, max(float(similarities[n]) for n in neighbors),
            )
            pairs.append((owner, doc))
            values.append(features_row)
    return pairs, np.asarray(values, dtype=float)


def fit(x, y):
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = np.column_stack((np.ones(len(x)), (x - center) / scale))
    beta = np.zeros(x.shape[1])
    penalty = np.diag([0.] + [5.] * (x.shape[1] - 1))
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        hessian = (x.T * (p * (1 - p))) @ x + penalty
        update = np.linalg.solve(hessian, x.T @ (p - y) + penalty @ beta)
        beta -= update
        if np.max(np.abs(update)) < 1e-7:
            break
    return center, scale, beta


def estimate(model, values):
    center, scale, beta = model
    return 1 / (1 + np.exp(-np.clip(beta[0] + ((values - center) / scale) @ beta[1:], -35, 35)))


def expected_doc(grouped, scores, people, lookahead):
    owner_index = {owner: i for i, owner in enumerate(people)}
    docs = sorted(grouped)
    probabilities = np.zeros((len(docs), len(people)))
    for i, doc in enumerate(docs):
        for owner, _ in grouped[doc]:
            probabilities[i, owner_index[owner]] = scores[owner, doc]
    expected = probabilities.sum(axis=1)
    order = sorted(range(len(docs)), key=lambda i: (-expected[i], docs[i]))
    if not lookahead or len(order) == 1:
        return docs[order[0]]
    pool = np.asarray(order[:LOOKAHEAD_WIDTH])
    selected = probabilities[pool]
    # Ordered pairs of reads give the same two-source coverage. Break ties
    # by the stronger first read before acquiring feedback.
    gain = expected[pool, None] + (1-selected) @ selected.T
    np.fill_diagonal(gain, -np.inf)
    pick = max(range(len(pool)), key=lambda i: (gain[i].max(), expected[pool[i]], -i))
    return docs[pool[pick]]


def schedule(people, scores, source_labels, expert_labels, cap, lookahead, pair_cap=PAIR_CAP, *, trace=None):
    rank = {owner: i for i, owner in enumerate(people)}
    opened, checked, supported = set(), set(), set()
    unknown = 0
    while len(checked) < pair_cap:
        eligible = [pair for pair in scores if pair[0] not in supported and pair not in checked]
        active = [pair for pair in eligible if pair[1] in opened]
        if not active:
            if not eligible or len(opened) >= cap:
                break
            grouped = defaultdict(list)
            for pair in eligible:
                grouped[pair[1]].append(pair)
            doc = expected_doc(grouped, scores, people, lookahead and len(opened)+1 < cap)
            active = grouped[doc]
        action = min(active, key=lambda p: (-scores[p], rank[p[0]], p[1]))
        owner, doc = action
        opened.add(doc)
        checked.add(action)
        if trace is not None:
            trace.append(action)
        mark = source_labels.get(owner, {}).get(doc)
        unknown += mark is None
        if mark == 2:
            supported.add(owner)
    issued = [owner for owner in people if owner in supported][:5]
    return dict(checks=len(checked), reads=len(opened), unknown=unknown, issued=len(issued),
                supported=sum(expert_labels.get(owner) == 2 for owner in issued))


def observations_for(index, topic, people, owned, experts, documents):
    pairs, matrix = features(index, topic, people, owned)
    truth = np.fromiter((experts[topic].get(owner) == 2 and documents[topic].get(owner, {}).get(doc) == 2
                        for owner, doc in pairs), dtype=float)
    return pairs, matrix, truth


def run(qrels, output, reverse):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run in a Slurm allocation with a fresh path")
    experts, documents = load(qrels)
    topics = sorted(experts, key=int)
    vectors = lexical_vectors(topics, read_topics(qrels / "ent06.expert.topics"))
    results = {}
    for number, held in enumerate(reversed(topics) if reverse else topics, 1):
        # Target labels are absent even when constructing every training
        # example's menus. Each example separately masks its own topic.
        history = HistoricalIndex({q: data for q,data in documents.items() if q != held}, topics, vectors)
        design, target = [], []
        for q in topics:
            if q == held:
                continue
            people, owned = expert_votes(history, history.positions[q])
            _, x, y = observations_for(history, q, people[:PERSON_CAP], owned, experts, documents)
            design.append(x)
            target.append(y)
        y = np.concatenate(target)
        model = fit(np.concatenate(design), y)
        panels = nominations(history, held)
        people, owned = panels["expert_first"]
        people = people[:PERSON_CAP]
        doc_rank, doc_sources = panels["document_first"]
        reference = allocate(doc_rank[:PERSON_CAP], doc_sources, documents[held], experts[held], 3, "rank_first")
        cap = reference["reads"]
        pairs, design, labels = observations_for(history, held, people, owned, experts, documents)
        p = estimate(model, design)
        scores = dict(zip(pairs, map(float, p), strict=True))
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite witness prediction")
        variant = {
            "reuse_reference": allocate(people, owned, documents[held], experts[held], cap, "reuse_first"),
            "fitted_greedy": schedule(people, scores, documents[held], experts[held], cap, False),
            "fitted_lookahead": schedule(people, scores, documents[held], experts[held], cap, True),
        }
        baseline = variant["reuse_reference"]
        strict = {"reuse_reference": baseline}
        for arm in ARMS[1:]:
            strict[arm] = schedule(people, scores, documents[held], experts[held],
                                   baseline["reads"], arm == "fitted_lookahead", baseline["checks"])
        if any(row["reads"] > baseline["reads"] or row["checks"] > baseline["checks"] for row in strict.values()):
            raise ValueError("A strict route exceeds realized reference work")
        known_support = {person: {doc for doc, label in refs.items() if label == 2}
                         for person, refs in documents[held].items() if experts[held][person] == 2}
        ceiling = oracle_ceiling(people, owned, known_support, cap)
        if any(row["reads"] > cap or row["checks"] > PAIR_CAP or row["supported"] > ceiling for row in variant.values()):
            raise ValueError("Comparison exceeds fixed budget or ceiling")
        results[held] = dict(arms=variant, both_caps_matched=strict, ceiling=ceiling, cap=cap,
            observed_labeled_pairs=int(labels.sum()), pairs=len(labels),
            squared_error=float(np.square(p-labels).sum()),
            frequency_baseline_squared_error=float(np.square(y.mean()-labels).sum()))
        if number % 10 == 0:
            print(f"Audited {number} held topics", flush=True)
    ordered = [results[q] for q in topics]
    summaries = {}
    for budget in ("arms", "both_caps_matched"):
        reference = [case[budget]["reuse_reference"] for case in ordered]
        summary = {}
        for arm in ARMS:
            current = [case[budget][arm] for case in ordered]
            summary[arm] = dict(
                **{key: sum(case[key] for case in current) for key in current[0]},
                paired_supported_pp5_delta=paired_interval([
                    20 * (after["supported"] - before["supported"])
                    for after,before in zip(current, reference, strict=True)]))
        summaries[budget] = summary
    stable = {q: {k: v for k, v in row.items() if not k.endswith("squared_error")} for q,row in results.items()}
    write_json(output, dict(topic_count=len(topics), reverse=reverse, pair_cap=PAIR_CAP,
        known_support_ceiling=sum(case["ceiling"] for case in ordered),
        total_reference_read_cap=sum(case["cap"] for case in ordered), **summaries,
        calibration_retrieved_menus={k: sum(row[k] for row in ordered) for k in (
            "pairs", "observed_labeled_pairs", "squared_error", "frequency_baseline_squared_error")},
        comparison_signature=sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(),
        inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256},
        runner_sha256=digest(Path(__file__)), slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("qrels", "output"):
        cli.add_argument(key, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
