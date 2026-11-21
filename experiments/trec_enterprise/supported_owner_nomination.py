"""Learn expert nomination for reachable evidence under a fixed verification budget."""

import argparse
from collections import Counter
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
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, fit
from experiments.trec_enterprise.nomination_read_budget import DEPTH, allocate, oracle_ceiling
from experiments.trec_enterprise.nomination_waterfall import PERSON_CAP, expert_votes, lexical_vectors, nominations

UNIVERSE = 60
CORE_SIZE = 15
SHARED_SOURCE_DEPTH = 4


def candidate_features(index, topic, order, sources):
    held = index.positions[topic]
    historical = ~(1 << held)
    similarities = np.maximum(index.similarity[held], 0.)
    profile = []
    for rank, owner in enumerate(order, 1):
        menu = sources[owner][:DEPTH]
        links = 0
        shared = []
        top_docs = []
        for doc, weight in menu:
            owners = index.support[doc]
            mask = owners[owner] & historical
            if not mask:
                raise ValueError("A candidate source depends on a held label")
            links |= mask
            shared.append(sum(bool(value & historical) for value in owners.values()))
            top_docs.append(weight)
        if not links:
            raise ValueError("Expected historical owners with at least one source")
        matched = (i for i in range(len(similarities)) if links & (1 << i))
        best_match = max(float(similarities[i]) for i in matched)
        profile.append([
            math.log(rank), math.log1p(len(sources[owner])), math.log1p(links.bit_count()),
            best_match, top_docs[0], sum(top_docs[:3]) / min(len(top_docs), 3),
            float(np.mean(shared)), math.log1p(shared[0]),
        ])
    return np.asarray(profile)


def targets(topic, people, sources, experts, documents):
    relevance = np.fromiter((experts[topic].get(owner) == 2 for owner in people), dtype=float)
    access = np.fromiter((bool(grade) and any(
        documents[topic].get(owner, {}).get(doc) == 2 for doc, _ in sources[owner][:DEPTH]
    ) for owner, grade in zip(people, relevance, strict=True)), dtype=float)
    return relevance, access


def learn(index, topics, held, experts, documents):
    x, relevance, access = [], [], []
    for topic in topics:
        if topic == held:
            continue
        people, sources = expert_votes(index, index.positions[topic])
        owners = people[:UNIVERSE]
        x.append(candidate_features(index, topic, owners, sources))
        r, a = targets(topic, owners, sources, experts, documents)
        relevance.append(r)
        access.append(a)
    matrix = np.concatenate(x)
    return fit(matrix, np.concatenate(relevance)), fit(matrix, np.concatenate(access))


def sorted_people(people, scores):
    return [people[i] for i in np.argsort(-scores, kind="stable")]


def reserved_people(people, scores, sources, require_reuse):
    core = people[:CORE_SIZE]
    core_sources = {doc for owner in core for doc, _ in sources[owner][:SHARED_SOURCE_DEPTH]}
    remaining = sorted_people(people[CORE_SIZE:], scores[CORE_SIZE:])
    if require_reuse:
        # Prefer additions with an independently ranked document in the core's
        # shallow menus. Preserve ranked fallbacks when fewer than five qualify.
        remaining.sort(key=lambda owner: not any(doc in core_sources
                           for doc, _ in sources[owner][:SHARED_SOURCE_DEPTH]))
    return core + remaining[:PERSON_CAP-CORE_SIZE]


def known_support(topic, experts, documents):
    return {owner: {doc for doc, g in rows.items() if g == 2}
            for owner, rows in documents[topic].items()
            if experts[topic].get(owner) == 2 and 2 in rows.values()}


def measure(people, sources, truth, expert_grade, source_grade, cap, checks, reference):
    trace = {}
    result = allocate(people, sources, source_grade, expert_grade, cap, "reuse_first", pair_cap=checks, trace=trace)
    result["supported_added_nominees"] = sum(
        expert_grade.get(owner) == 2 and owner not in reference for owner in trace["issued_owners"]
    )
    result["grade2_experts"] = sum(expert_grade.get(owner) == 2 for owner in people)
    result["expert_with_historical_source"] = sum(bool(truth.get(owner, set()).intersection(
        doc for doc, _ in sources[owner][:DEPTH])) for owner in people)
    result["oracle_supported_slots"] = oracle_ceiling(people, sources, truth, cap)
    if result["supported"] > result["oracle_supported_slots"] or result["reads"] > cap or result["checks"] > checks:
        raise ValueError("Outcome exceeds the matched budget or disclosed ceiling")
    return result


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run in a Slurm allocation and use a fresh output")
    experts, documents = load(args.qrels)
    topics = sorted(experts, key=int)
    vectors = lexical_vectors(topics, read_topics(args.qrels / "ent06.expert.topics"))
    rows = {}
    traversal = topics[::-1] if args.reverse else topics
    for j, held in enumerate(traversal, 1):
        history = HistoricalIndex({key: entry for key, entry in documents.items() if key != held}, topics, vectors)
        ranker, witness_ranker = learn(history, topics, held, experts, documents)
        nominees = nominations(history, held)
        whole, sources = nominees["expert_first"]
        people = whole[:UNIVERSE]
        standard = people[:PERSON_CAP]
        matrix = candidate_features(history, held, people, sources)
        topical = estimate(ranker, matrix)
        supported = estimate(witness_ranker, matrix)
        arms = {
            "expert_first": standard,
            "learned_topic": sorted_people(people, topical)[:PERSON_CAP],
            "learned_witness_reorder": sorted_people(standard, supported[:PERSON_CAP]),
            "learned_witness_expand": sorted_people(people, supported)[:PERSON_CAP],
            "reserved_witness": reserved_people(people, supported, sources, False),
            "reserved_witness_shared": reserved_people(people, supported, sources, True),
        }
        opening_people, opening_sources = nominees["document_first"]
        opening = allocate(opening_people[:PERSON_CAP], opening_sources,
                           documents[held], experts[held], 3, "rank_first")
        base = allocate(standard, sources, documents[held], experts[held], opening["reads"], "reuse_first")
        cap, checks = base["reads"], base["checks"]
        truth = known_support(held, experts, documents)
        measured = {}
        for name, order in arms.items():
            measured[name] = measure(order, sources, truth, experts[held], documents[held], cap, checks, set(standard))
            measured[name]["replaced_nominees"] = len(set(order) - set(standard))
        if any(measured["expert_first"][key] != value for key, value in base.items()):
            raise ValueError("Replay changed the earlier expert-first reference")
        rows[held] = measured
        if j % 10 == 0:
            print(f"Evaluated {j} held topics", flush=True)
    summary = {}
    comparisons = ("expert_first", "learned_topic")
    for name in arms:
        outcomes = [rows[topic][name] for topic in topics]
        contrasts = {}
        for base in comparisons:
            reference = [rows[topic][base] for topic in topics]
            contrasts[base] = {
                "supported_p_at_5_pp": paired_interval([
                    20 * (a["supported"] - b["supported"]) for a, b in zip(outcomes, reference, strict=True)
                ]),
                "reachable_experts_per_topic": paired_interval([
                    a["expert_with_historical_source"] - b["expert_with_historical_source"]
                    for a, b in zip(outcomes, reference, strict=True)
                ]),
            }
        totals = sum((Counter(row) for row in outcomes), Counter())
        summary[name] = dict.fromkeys(outcomes[0], 0) | dict(totals) | {"contrasts": contrasts}
    signature = sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    write_json(args.output, dict(results=summary, candidates=PERSON_CAP, training_universe=UNIVERSE,
                                topic_count=len(topics), reverse=args.reverse, comparison_signature=signature,
                                reserved_core=CORE_SIZE, shared_source_depth=SHARED_SOURCE_DEPTH,
                                inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256},
                                runner=digest(Path(__file__)), slurm=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("qrels", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
