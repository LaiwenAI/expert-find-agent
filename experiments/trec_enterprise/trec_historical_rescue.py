"""Compare candidate and citation expansion without using held-topic pool membership."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import math
import os
from pathlib import Path

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.credit import live_joint_source_request as live
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.followir.core17_reranking import digest, write_json


PEOPLE = 20
DEPTH = 256
CAPS = (2, 4, 8)
SEEDS = tuple(range(12))
POLICIES = ("deep", "wide", "alternating", "value")
VIEWS = ("all_historical", "positive_historical")
VERIFIERS = {"judged_support": (1., 0.), "correlated_error": (.9, .05)}


def prepare(index, topic, view):
    pos = index.positions[topic]
    held = 1 << pos
    ranks = index.ranking(topic, {pos})
    graph = index.support if view == "positive_historical" else index.owners
    members = index.contributor_sets((ranks,), {pos}, support_only=view == "positive_historical")
    sources = defaultdict(list)
    for document, rank in ranks.items():
        for owner in members[document]:
            historical = graph[document][owner] & ~held
            positive = (index.support.get(document, {}).get(owner, 0) & ~held).bit_count()
            # Bound document influence and retain ability to select an ungraded historical link.
            chance = (positive + .25) / (historical.bit_count() + 1)
            contribution = chance / math.sqrt(rank)
            sources[owner].append((document, contribution, positive))
    scores = {owner: sum(sorted((rank for _, rank, _ in values), reverse=True)[:3])
              for owner, values in sources.items()}
    order = sorted(sources, key=lambda owner: (-scores[owner], owner))[:PEOPLE]
    details = {owner: sorted(sources[owner], key=lambda row: (-row[1], row[0])) for owner in order}
    return order, details


def number(seed, name):
    data = f"{seed}:{name}".encode()
    return int.from_bytes(sha256(data).digest()[:8], "big") / 2**64


def read(pair, source_labels, noise, seed):
    owner, doc = pair
    grade = source_labels.get(owner, {}).get(doc)
    if grade is None:
        return False, None
    sensitivity, false_positive = noise
    flag = number(seed, doc) < (sensitivity if grade == 2 else false_positive)
    return flag, grade


def metrics(observed, candidates, labels):
    ranks = {owner: i for i, owner in enumerate(candidates)}
    matches = {}
    for (owner, doc), (accepted, grade) in observed.items():
        if accepted:
            matches.setdefault(owner, (doc, grade))
    issued = sorted(matches, key=ranks.__getitem__)[:5]
    grounded = sum(labels.get(owner) == 2 and matches[owner][1] == 2 for owner in issued)
    unknown = sum(owner not in labels for owner in issued)
    return dict(
        issued=len(issued), grounded=grounded, adverse_known=len(issued)-grounded-unknown,
        issued_unknown_expert=unknown, pure_expert=sum(labels.get(owner) == 2 for owner in issued),
        checked=len(observed), distinct_documents=len({doc for _, doc in observed}),
        unknown_checks=sum(grade is None for _, grade in observed.values()),
    )


def choose(frontier, observed, rank, policy, step):
    supported = {owner for (owner, _), (ok, _) in observed.items() if ok}
    available = [(owner, doc, value) for owner, doc, value in frontier if (owner, doc) not in observed]
    outside = lambda owner: rank[owner] >= 5
    if policy == "deep":
        acceptable = [row for row in available if not outside(row[0])]
    elif policy == "wide":
        acceptable = [row for row in available if outside(row[0])]
    elif policy == "alternating":
        acceptable = [row for row in available if outside(row[0]) == (step % 2 == 0)] or available
    else:
        acceptable = available
    if not acceptable:
        return None
    owner_reads = Counter(owner for owner, _ in observed)
    best = min(acceptable, key=lambda row: (
        row[0] in supported, -row[2] / (1 + .35 * owner_reads[row[0]]), rank[row[0]], row[1],
    ))
    return best[:2]


def run_case(order, details, expert_labels, source_labels, cap, noise, seed, scope):
    ranks = {owner: i for i, owner in enumerate(order)}
    initial = {(owner, details[owner][0][0]): read((owner, details[owner][0][0]), source_labels, noise, seed)
               for owner in order[:5] if details[owner]}
    base = metrics(initial, order, expert_labels)
    frontier = [(owner, doc, value) for owner in order[:scope] for doc, value, _ in details[owner]]
    results = {}
    for policy in POLICIES:
        observed = initial.copy()
        for step in range(cap):
            action = choose(frontier, observed, ranks, policy, step)
            if action is None:
                break
            observed[action] = read(action, source_labels, noise, seed)
        result = metrics(observed, order, expert_labels)
        result["extra_checks"] = len(observed) - len(initial)
        result["recovered_after_abstention"] = (not base["issued"]) and bool(result["grounded"])
        results[policy] = result
    return base, results



def reachable_support(prepared, experts, documents):
    report = Counter()
    for topic, (order, details) in prepared.items():
        labels = experts[topic]
        source_labels = documents[topic]
        known = {owner for owner, grade in labels.items() if grade == 2}
        visible = {owner for owner in order if owner in known and any(
            source_labels.get(owner, {}).get(doc) == 2 for doc, _, _ in details[owner]
        )}
        inside = set(order[:5]) & visible
        within_ten = set(order[:10]) & visible
        outside = visible - inside
        report.update(
            topics=1, known_experts=len(known), reachable_within_five=len(inside),
            reachable_within_ten=len(within_ten), reachable_within_twenty=len(visible),
            topics_with_support_within_five=bool(inside),
            topics_with_support_within_twenty=bool(visible),
            topics_with_support_only_outside_five=bool(outside and not inside),
        )
    return dict(report)

def study(index, vectors, experts, documents):
    output = {}
    for view in VIEWS:
        prepared = {topic: prepare(index, topic, view) for topic in index.topics}
        alt_docs = {key: value for key, value in documents.items() if key != index.topics[0]}
        alt_index = HistoricalIndex(alt_docs, index.topics, vectors)
        if prepare(alt_index, index.topics[0], view) != prepared[index.topics[0]]:
            raise AssertionError("Candidate discovery depended on held-topic judgment membership")
        output[view] = {"reachable_support": reachable_support(prepared, experts, documents)}
        for noise_name, noise in VERIFIERS.items():
            output[view][noise_name] = {}
            for scope in (10, 20):
                for cap in CAPS:
                    rows = {arm: [] for arm in POLICIES}
                    initial_totals = Counter()
                    matched = defaultdict(list)
                    for topic, (order, details) in prepared.items():
                        group = {arm: Counter() for arm in POLICIES}
                        match = {arm: [] for arm in POLICIES}
                        for jitter in SEEDS if noise_name != "judged_support" else (0,):
                            base, outcomes = run_case(order, details, experts[topic], documents[topic], cap, noise,
                                                      f"{topic}:{jitter}", scope)
                            initial_totals.update(base)
                            initial_totals.update(replays=1, initial_abstention=not base["issued"])
                            deep = outcomes["deep"]
                            for arm, result in outcomes.items():
                                group[arm].update(result)
                                if result["extra_checks"] == deep["extra_checks"]:
                                    match[arm].append((result["grounded"] - deep["grounded"]) / 5)
                        for arm in POLICIES:
                            n = len(SEEDS) if noise_name != "judged_support" else 1
                            rows[arm].append({key: value / n for key, value in group[arm].items()})
                            if match[arm]:
                                matched[arm].append(float(np.mean(match[arm])))
                    summary = {}
                    for arm, entries in rows.items():
                        summary[arm] = {
                            **{key: sum(row[key] for row in entries) for key in entries[0]},
                            "all_topic_delta_pp5": paired_interval([
                                (row["grounded"] - prior["grounded"]) / 5
                                for row, prior in zip(entries, rows["deep"], strict=True)
                            ]),
                            "matched_checks_delta_pp5": paired_interval(matched[arm]) if matched[arm] else None,
                            "topics_with_matched": len(matched[arm]),
                        }
                    output[view][noise_name][f"scope{scope}_cap{cap}"] = dict(initial=initial_totals, policies=summary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qrels", "embedding", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm allocation and a fresh output")
    checkpoint = live.verify_embedding(args.embedding)
    experts, documents = load(args.qrels)
    topics = sorted(experts, key=int)
    text = read_topics(args.qrels / "ent06.expert.topics")
    vectors = QwenTorchEmbedding(args.embedding).encode([text[topic] for topic in topics], query=True)
    index = HistoricalIndex(documents, topics, vectors)
    write_json(args.output, dict(topics=len(topics), source_depth=DEPTH, historical_documents=len(index.ids),
                                 checkpoint=checkpoint, input_sha256={**FILES, "ent06.expert.topics":TOPIC_SHA256},
                                 assumed_error_profiles=VERIFIERS, results=study(index, vectors, experts, documents)))


if __name__ == "__main__":
    main()
