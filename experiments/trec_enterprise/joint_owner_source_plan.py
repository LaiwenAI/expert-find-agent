"""Plan expert nominations around the source bundles that fit the read budget."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, features, fit, observations_for, schedule
from experiments.trec_enterprise.nomination_read_budget import allocate, oracle_ceiling
from experiments.trec_enterprise.nomination_waterfall import PERSON_CAP, expert_votes, lexical_vectors, nominations
from experiments.trec_enterprise.supported_owner_nomination import (
    UNIVERSE, candidate_features, known_support, sorted_people, targets,
)


def fit_models(index, topics, held, experts, documents):
    pair_rows, pair_truth, people_rows, people_truth = [], [], [], []
    for topic in topics:
        if topic == held:
            continue
        people, sources = expert_votes(index, index.positions[topic])
        people = people[:UNIVERSE]
        _, x, y = observations_for(index, topic, people, sources, experts, documents)
        pair_rows.append(x)
        pair_truth.append(y)
        people_rows.append(candidate_features(index, topic, people, sources))
        people_truth.append(targets(topic, people, sources, experts, documents)[1])
    matrix, values = np.concatenate(pair_rows), np.concatenate(pair_truth)
    original = matrix[:, 0] <= math.log(PERSON_CAP)
    return (
        fit(matrix[original], values[original]),
        fit(matrix, values),
        fit(np.concatenate(people_rows), np.concatenate(people_truth)),
    )


def nominate_plan(people, scores, read_cap, check_cap):
    """Reserve a balanced check quota for each unopened source."""
    grouped = defaultdict(dict)
    for (owner, doc), score in scores.items():
        grouped[doc][owner] = score
    rank = {owner: i for i, owner in enumerate(people)}
    selected_docs, checked_people = [], {}
    residual = dict.fromkeys(people, 1.)
    remaining_checks = check_cap
    for wave in range(read_cap):
        if not grouped or not remaining_checks:
            break
        quota = math.ceil(remaining_checks / (read_cap - wave))
        candidates = {}
        for doc, owners in grouped.items():
            order = sorted(owners, key=lambda owner: (-owners[owner] * residual[owner], rank[owner]))
            subset = order[:quota]
            gain = sum(owners[owner] * residual[owner] for owner in subset)
            candidates[doc] = (gain, subset)
        best_doc = min(candidates, key=lambda doc: (-candidates[doc][0], doc))
        if not candidates[best_doc][0]:
            break
        selected_docs.append(best_doc)
        owners = grouped.pop(best_doc)
        proposed = candidates[best_doc][1]
        for owner in proposed:
            residual[owner] *= 1 - owners[owner]
            checked_people[owner] = None
        remaining_checks -= len(proposed)
    nominees = list(dict.fromkeys([*checked_people, *people]))[:PERSON_CAP]
    return nominees, set(selected_docs), len(checked_people)


def adaptive_plan(people, scores, source_labels, expert_labels, read_cap, check_cap):
    """Choose a new source only when its check quota competes with current reads."""
    order = {owner: i for i, owner in enumerate(people)}
    opened, examined, supported = set(), set(), set()
    admitted = {}
    unknown = 0
    while len(examined) < check_cap:
        active, unseen = [], defaultdict(list)
        for pair in scores:
            owner, doc = pair
            if pair in examined or owner in supported:
                continue
            if doc in opened:
                active.append(pair)
            elif len(opened) < read_cap:
                unseen[doc].append(pair)
        preference = lambda pair: (-scores[pair], order[pair[0]], pair[1])
        best_active = min(active, key=preference, default=None)
        best_new = None
        if unseen:
            quota = math.ceil((check_cap-len(examined)) / (read_cap-len(opened)))
            bundles = {doc: sorted(pairs, key=preference)[:quota] for doc, pairs in unseen.items()}
            strengths = {doc: sum(scores[pair] for pair in pairs) / quota for doc, pairs in bundles.items()}
            best_doc = min(strengths, key=lambda doc: (-strengths[doc], doc))
            if best_active is None or strengths[best_doc] > scores[best_active]:
                best_new = bundles[best_doc][0]
        action = best_new or best_active
        if action is None:
            break
        owner, doc = action
        opened.add(doc)
        examined.add(action)
        admitted[owner] = None
        grade = source_labels.get(owner, {}).get(doc)
        unknown += grade is None
        if grade == 2:
            supported.add(owner)

    if len(admitted) > PERSON_CAP:
        raise ValueError("Adaptive admission exceeds the shortlist cap")
    nominees = list(dict.fromkeys([*admitted, *people]))[:PERSON_CAP]
    issued = [owner for owner in nominees if owner in supported][:5]
    return nominees, dict(reads=len(opened), checks=len(examined), unknown=unknown, issued=len(issued),
                           supported=sum(expert_labels.get(owner) == 2 for owner in issued))


def subset(scores, owners, docs=None):
    owners = set(owners)
    return {key: p for key, p in scores.items() if key[0] in owners and (docs is None or key[1] in docs)}


def add_executed(result, selected, truth, sources, cap, checks, baseline):
    if result["reads"] > cap or result["checks"] > checks:
        raise ValueError("A route exceeded its matched read/check cap")
    result["oracle_ceiling"] = oracle_ceiling(selected, sources, truth, cap)
    if result["supported"] > result["oracle_ceiling"]:
        raise ValueError("Observed support exceeds the read oracle")
    result["changed_nominees"] = len(set(selected) - set(baseline))
    return result


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run on a Slurm compute allocation with a fresh output path")
    experts, evidence = load(args.qrels)
    topics = sorted(experts, key=int)
    text = read_topics(args.qrels / "ent06.expert.topics")
    vectors = lexical_vectors(topics, text)
    records, timing = {}, defaultdict(list)
    traversal = topics[::-1] if args.reverse else topics
    for i, held in enumerate(traversal, 1):
        graph = HistoricalIndex({key: value for key, value in evidence.items() if key != held}, topics, vectors)
        legacy, pair_model, owner_model = fit_models(graph, topics, held, experts, evidence)
        sections = nominations(graph, held)
        people, sources = sections["expert_first"]
        people, original = people[:UNIVERSE], people[:PERSON_CAP]
        doc_people, doc_sources = sections["document_first"]
        source_labels, expert_labels = evidence[held], experts[held]
        starter = allocate(doc_people[:PERSON_CAP], doc_sources, source_labels, expert_labels, 3, "rank_first")
        base = allocate(original, sources, source_labels, expert_labels, starter["reads"], "reuse_first")
        cap, checks = base["reads"], base["checks"]
        truth = known_support(held, experts, evidence)
        # Source and owner features are a fixed view of the original expert-first rank.
        pairs, design = features(graph, held, people, sources)
        score = dict(zip(pairs, map(float, estimate(pair_model, design)), strict=True))
        observed_original = subset(score, original)
        legacy_pairs, legacy_design = features(graph, held, original, sources)
        legacy_score = dict(zip(legacy_pairs, map(float, estimate(legacy, legacy_design)), strict=True))
        predicted_owner = estimate(owner_model, candidate_features(graph, held, people, sources))
        expanded = sorted_people(people, predicted_owner)[:PERSON_CAP]
        rows = {
            "reuse_reference": add_executed(base, original, truth, sources, cap, checks, original),
        }
        for name, selected, source_scores in (
            ("fitted_legacy", original, legacy_score),
            ("fitted_wide_fixed", original, observed_original),
            ("fitted_wide_expanded", expanded, subset(score, expanded)),
        ):
            result = schedule(selected, source_scores, source_labels, expert_labels, cap, False, checks)
            rows[name] = add_executed(result, selected, truth, sources, cap, checks, original)

        for name, menu, source_scores in (
            ("planned_fixed", original, observed_original),
            ("planned_joint", people, score),
        ):
            started = perf_counter()
            selected, docs, planned_people = nominate_plan(menu, source_scores, cap, checks)
            available = subset(source_scores, selected, docs)
            timing[name].append(1_000 * (perf_counter() - started))
            result = schedule(selected, available, source_labels, expert_labels, cap, False, checks)
            outcome = add_executed(result, selected, truth, sources, cap, checks, original)
            outcome["planned_docs"] = len(docs)
            outcome["planned_distinct_owners"] = planned_people
            outcome["oracle_inside_planned_sources"] = min(5, sum(any(
                (owner, doc) in available for doc in truth.get(owner, ())
            ) for owner in selected))
            rows[name] = outcome
            if name == "planned_joint":
                open_result = schedule(selected, subset(score, selected), source_labels, expert_labels, cap, False, checks)
                rows["joint_nominees_open_sources"] = add_executed(
                    open_result, selected, truth, sources, cap, checks, original
                )

        for name, menu, source_scores in (("adaptive_fixed", original, observed_original),
                                           ("adaptive_joint", people, score)):
            started = perf_counter()
            selected, result = adaptive_plan(menu, source_scores, source_labels, expert_labels, cap, checks)
            timing[name].append(1_000 * (perf_counter() - started))
            rows[name] = add_executed(result, selected, truth, sources, cap, checks, original)
        records[held] = rows
        if i % 10 == 0:
            print(f"Completed {i} held topics", flush=True)

    summaries = {}
    for name in rows:
        results = [records[topic][name] for topic in topics]
        totals = sum((Counter(result) for result in results), Counter())
        comparison = {
            reference: paired_interval([
                20 * (records[topic][name]["supported"] - records[topic][reference]["supported"])
                for topic in topics
            ])
            for reference in ("reuse_reference", "fitted_legacy", "planned_fixed")
        }
        summaries[name] = dict.fromkeys(results[0], 0) | dict(totals) | {"paired_p_at_5_pp": comparison}
    write_json(args.output, dict(
        results=summaries, reverse=args.reverse, topic_count=len(topics), nomination_universe=UNIVERSE,
        comparison_signature=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        policy_cpu_p95_ms={name: float(np.quantile(values, .95)) for name, values in timing.items()},
        inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256}, runner=digest(Path(__file__)),
        slurm=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("qrels", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
