"""Replay one-citation rescue against person-specific TREC document grades."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load
from experiments.trec_enterprise.ownership_metric_audit import selected_ranking


TRIALS = 32
FRONTIER = 4
ARMS = ("next_checked", "topic_only", "topic_shared_checked", "topic_all_checked",
        "topic_shared_backfill", "topic_all_backfill", "sequential_checked", "oracle")


def document_views(documents):
    indexed = defaultdict(dict)
    for expert, judgments in documents.items():
        for source, grade in judgments.items():
            indexed[source][expert] = grade
    return indexed


def choose(arm, extra, expert, indexed, topical):
    if not extra:
        return None, 0
    if arm == "next_checked":
        candidate = extra[0]
        return (candidate if indexed[candidate][expert] == 2 else None), 1
    if arm == "sequential_checked":
        for count, candidate in enumerate(extra, start=1):
            if indexed[candidate][expert] == 2:
                return candidate, count
        return None, len(extra)
    if arm == "oracle":
        return next((doc for doc in extra if indexed[doc][expert] == 2), None), len(extra)
    checks = 0
    for doc in extra:
        judgments = indexed[doc]
        if not topical(doc):
            continue
        shared = arm in ("topic_shared_checked", "topic_shared_backfill") and len(judgments) > 1
        if shared or arm in ("topic_all_checked", "topic_all_backfill"):
            checks += 1
            if judgments[expert] < 2:
                if arm in ("topic_shared_backfill", "topic_all_backfill"):
                    continue
                return None, checks
        return doc, checks
    return None, checks


def run_trial(ranks, grades, documents, indexed, topic, trial, noisy):
    report = {arm: Counter() for arm in ARMS}
    inventory = Counter()
    noise = {}

    def topic_has_support(doc):
        return 2 in indexed[doc].values()

    def topical(doc):
        if not noisy:
            return topic_has_support(doc)
        if doc not in noise:
            draw = random.Random(f"trec-rescue-v1:topical:{topic}:{trial}:{doc}").random()
            noise[doc] = draw < (0.90 if topic_has_support(doc) else 0.05)
        return noise[doc]

    for expert in ranks:
        sources = sorted(documents.get(expert, {}))
        random.Random(f"trec-rescue-v1:{topic}:{trial}:{expert}").shuffle(sources)
        front, extra = sources[:3], sources[3:3 + FRONTIER]
        # A perfect first-stage verifier supplies exactly one supporting anchor.
        anchors = [doc for doc in front if indexed[doc][expert] == 2]
        if len(anchors) != 1 or not extra:
            continue
        inventory["eligible"] += 1
        relevant = grades[expert] == 2
        inventory["eligible_relevant"] += relevant
        inventory["in_pool_valid"] += any(indexed[doc][expert] == 2 for doc in extra)
        if any(2 in indexed[doc].values() and indexed[doc][expert] < 2 for doc in extra):
            inventory["has_borrowed_support"] += 1
        for arm in ARMS:
            doc, checks = choose(arm, extra, expert, indexed, topical)
            report[arm]["personal_checks"] += checks
            if doc is None:
                continue
            grade = indexed[doc][expert]
            report[arm]["attached"] += 1
            report[arm]["valid_pairs"] += grade == 2
            report[arm]["clean_relevant_pairs"] += grade == 2 and relevant
            report[arm]["neutral_additions"] += grade == 1
            report[arm]["contradictory_additions"] += grade == 0
            report[arm]["topic_false_acceptances"] += not topic_has_support(doc)
    return inventory, report


def summarize(topics):
    inventory = Counter()
    methods = {arm: Counter() for arm in ARMS}
    paired = {arm: [] for arm in ARMS}
    for trials in topics:
        for counts, row in trials:
            inventory.update(counts)
            for arm in ARMS:
                methods[arm].update(row[arm])
        baseline = mean(row["next_checked"]["clean_relevant_pairs"] for _, row in trials)
        for arm in ARMS:
            completions = mean(row[arm]["clean_relevant_pairs"] for _, row in trials)
            paired[arm].append((completions - baseline) / TOP_K)
    report = {}
    for arm, counts in methods.items():
        attached = counts["attached"]
        report[arm] = dict(counts, attachment_precision=counts["valid_pairs"] / attached,
                           clean_relevant_p5=counts["clean_relevant_pairs"] / (len(topics) * TRIALS * TOP_K),
                           checks_per_eligible=counts["personal_checks"] / inventory["eligible"],
                           clean_relevant_p5_delta_vs_next_checked=paired_interval(paired[arm]))
        for field in ("attached", "valid_pairs", "clean_relevant_pairs", "neutral_additions",
                      "contradictory_additions", "personal_checks", "topic_false_acceptances"):
            report[arm].setdefault(field, 0)
    return dict(inventory=inventory, arms=report)


def run(directory, output):
    experts, documents = load(directory)
    topics = []
    noisy_topics = []
    for topic in sorted(experts, key=int):
        grades = experts[topic]
        topic_docs = documents[topic]
        indexed = document_views(topic_docs)
        ranks = selected_ranking(experts, topic)
        topics.append([run_trial(ranks, grades, topic_docs, indexed, topic, i, False) for i in range(TRIALS)])
        noisy_topics.append([run_trial(ranks, grades, topic_docs, indexed, topic, i, True) for i in range(TRIALS)])
    result = dict(input_sha256=FILES, code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  topics=len(topics), trials_per_topic=TRIALS, extra_frontier=FRONTIER, **summarize(topics),
                  noisy_topic_simulation={"sensitivity": 0.90, "false_positive_rate": 0.05,
                                          **summarize(noisy_topics)})
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
