"""Test whether learning same-source contrasts improves expert verification."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, features, fit
from experiments.trec_enterprise.nomination_waterfall import PERSON_CAP, expert_votes, lexical_vectors
from experiments.trec_enterprise.supported_owner_nomination import UNIVERSE

READS = 3
CHECKS_PER_SOURCE = 2


def group_by_source(pairs):
    groups = defaultdict(list)
    for i, (_, doc) in enumerate(pairs):
        groups[doc].append(i)
    return groups


def observations(history, topic, experts, documents):
    owners, menus = expert_votes(history, history.positions[topic])
    owners = owners[:UNIVERSE]
    pairs, design = features(history, topic, owners, menus)
    relevance = np.array([experts[topic].get(owner, -1) for owner, _ in pairs])
    grades = np.array([documents[topic].get(owner, {}).get(doc, -1) for owner, doc in pairs])
    return owners, pairs, design, relevance, grades


def matched_pairs(groups, relevance, grades):
    return [
        (positive, negative)
        for indices in groups.values()
        for positive in indices if relevance[positive] == 2 and grades[positive] == 2
        for negative in indices if relevance[negative] == 2 and grades[negative] in (0, 1)
    ]


def fit_models(history, topics, held, experts, documents):
    menu_rows, menu_targets, judged_rows, judged_targets, differences = [], [], [], [], []
    for topic in topics:
        if topic == held:
            continue
        _, pairs, design, relevance, grades = observations(history, topic, experts, documents)
        strict = (relevance == 2) & (grades == 2)
        judged = (relevance == 2) & (grades >= 0)
        menu_rows.append(design)
        menu_targets.append(strict)
        judged_rows.append(design[judged])
        judged_targets.append(strict[judged])
        matches = matched_pairs(group_by_source(pairs), relevance, grades)
        if matches:
            positive, negative = np.asarray(matches).T
            differences.append(design[positive] - design[negative])
    differences = np.concatenate(differences)
    labels = np.r_[np.ones(len(differences)), np.zeros(len(differences))]
    return {
        "menu_pointwise": fit(np.concatenate(menu_rows), np.concatenate(menu_targets)),
        "judged_pointwise": fit(np.concatenate(judged_rows), np.concatenate(judged_targets)),
        "same_source_pairwise": fit(np.vstack((differences, -differences)), labels),
    }


def fixed_sources(owners, pairs, pointwise_score):
    rank = {owner: i for i, owner in enumerate(owners[:PERSON_CAP], 1)}
    indices = defaultdict(list)
    for i, (owner, doc) in enumerate(pairs):
        if owner in rank:
            indices[doc].append(i)
    prior = {doc: sum(1 / rank[pairs[i][0]] for i in rows) for doc, rows in indices.items()}
    expected = {
        doc: sum(sorted((float(pointwise_score[i]) for i in rows), reverse=True)[:CHECKS_PER_SOURCE])
        for doc, rows in indices.items()
    }
    return {
        name: [indices[doc] for doc in sorted(indices, key=lambda doc: (-priority[doc], doc))[:READS]]
        for name, priority in (("rank_sum", prior), ("expected_support", expected))
    }


def concordance(grades, matches, score):
    comparisons, negatives = Counter(), Counter()
    for positive, negative in matches:
        outcome = np.sign(score[positive] - score[negative])
        comparisons["wins"] += int(outcome > 0)
        comparisons["ties"] += int(outcome == 0)
        comparisons["losses"] += int(outcome < 0)
        if grades[negative] == 0:
            negatives["wins"] += int(outcome > 0)
            negatives["ties"] += int(outcome == 0)
            negatives["losses"] += int(outcome < 0)

    return dict(comparisons=dict(comparisons), grade_zero_comparisons=dict(negatives))


def evaluate(pairs, relevance, grades, reads, score, quota):
    checked = [i for indices in reads for i in sorted(indices, key=lambda i: (-score[i], i))[:quota]]
    is_support = (grades == 2) & (relevance == 2)
    confirmed = {pairs[i][0] for i in checked if is_support[i]}
    available = {pairs[i][0] for indices in reads for i in indices if is_support[i]}
    return dict(
        reads=len(reads), checks=len(checked), known_supported_people_available=min(5, len(available)),
        strict_pairs=int(is_support[checked].sum()), supported=min(5, len(confirmed)),
        negative_checks=int((grades[checked] == 0).sum()),
        neutral_checks=int((grades[checked] == 1).sum()),
        unknown_checks=int((grades[checked] == -1).sum()),
    )


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run in a Slurm allocation with a fresh path")
    experts, documents = load(args.qrels)
    topics = sorted(experts, key=int)
    vectors = lexical_vectors(topics, read_topics(args.qrels / "ent06.expert.topics"))
    records = {}
    for held in reversed(topics) if args.reverse else topics:
        history = HistoricalIndex({t: documents[t] for t in topics if t != held}, topics, vectors)
        models = fit_models(history, topics, held, experts, documents)
        owners, pairs, design, relevance, grades = observations(history, held, experts, documents)
        matches = matched_pairs(group_by_source(pairs), relevance, grades)
        scores = {"owner_rank": -design[:, 0], **{name: estimate(model, design) for name, model in models.items()}}
        panels = fixed_sources(owners, pairs, scores["menu_pointwise"])
        records[held] = {
            "concordance": {name: concordance(grades, matches, score) for name, score in scores.items()},
            **{
                f"{panel}_{quota}_checks": {
                    name: evaluate(pairs, relevance, grades, reads, score, quota) for name, score in scores.items()
                }
                for panel, reads in panels.items()
                for quota in (1, CHECKS_PER_SOURCE)
            },
        }

    result = {}
    for panel in records[topics[0]]:
        routes = {name: [records[t][panel][name] for t in topics] for name in scores}
        result[panel] = {}
        for name, rows in routes.items():
            if panel == "concordance":
                keys = ("comparisons", "grade_zero_comparisons")
                result[panel][name] = {key: dict(sum((Counter(row[key]) for row in rows), Counter())) for key in keys}
                continue
            tally = dict(sum((Counter(row) for row in rows), Counter()))
            tally["p_at_5_vs_owner_rank_pp"] = paired_interval([
                20 * (row["supported"] - prior["supported"])
                for row, prior in zip(rows, routes["owner_rank"], strict=True)
            ])
            result[panel][name] = tally
        if panel != "concordance":
            for key in ("reads", "checks", "known_supported_people_available"):
                if any([row[key] for row in rows] != [row[key] for row in routes["owner_rank"]] for rows in routes.values()):
                    raise ValueError(f"Unmatched {key} in {panel}")
    write_json(args.output, dict(
        results=result, topic_count=len(topics), reverse=args.reverse, slurm=os.environ["SLURM_JOB_ID"],
        topics_with_contrasts=sum(bool(records[t]["concordance"]["owner_rank"]["comparisons"]) for t in topics),
        signature=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256}, runner=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qrels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
