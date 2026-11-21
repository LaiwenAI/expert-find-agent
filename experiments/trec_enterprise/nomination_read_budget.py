"""Separate expert nomination from source reuse under two retrieval budgets."""

import argparse
from collections import Counter, defaultdict
from functools import reduce
from hashlib import sha256
from itertools import combinations
import json
import os
from operator import or_
from pathlib import Path

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.trec_enterprise.nomination_waterfall import ARMS, PERSON_CAP, lexical_vectors, nominations

DEPTH = 16
PAIR_CAP = 13
READ_CAPS = (2, 3, 5)
POLICIES = ("rank_first", "reuse_first")


def allocate(order, sources, source_labels, expert_labels, read_cap, policy, *, pair_cap=PAIR_CAP, trace=None):
    ranks = {owner: i for i, owner in enumerate(order, 1)}
    action_rank = {(owner, doc): rank for owner in order
                   for rank, (doc, _) in enumerate(sources[owner][:DEPTH], 1)}
    tested, opened, supported, attempted = set(), set(), set(), set()
    unknown = 0
    while len(tested) < pair_cap:
        remaining = [(owner, doc) for owner, doc in action_rank if owner not in supported
                     and (owner, doc) not in tested and (doc in opened or len(opened) < read_cap)]
        if not remaining:
            break
        if policy == "reuse_first":
            existing = [pair for pair in remaining if pair[1] in opened]
            if existing:
                remaining = existing
            else:
                grouped = defaultdict(list)
                for pair in remaining:
                    grouped[pair[1]].append(pair)
                strength = {doc: sum(1 / ((5 + ranks[owner]) * action_rank[owner, src])
                                     for owner, src in pairs) for doc, pairs in grouped.items()}
                doc = min(grouped, key=lambda d: (-strength[d], d))
                remaining = grouped[doc]
        owner, doc = min(remaining, key=lambda pair: (pair[0] in attempted, action_rank[pair],
                                                       ranks[pair[0]], pair[1]))
        attempted.add(owner)
        tested.add((owner, doc))
        opened.add(doc)
        mark = source_labels.get(owner, {}).get(doc)
        unknown += mark is None
        if mark == 2:
            supported.add(owner)
    issued = [owner for owner in order if owner in supported][:5]
    if trace is not None:
        trace["issued_owners"] = issued
    return dict(checks=len(tested), reads=len(opened), unknown=unknown, issued=len(issued),
                supported=sum(expert_labels.get(owner) == 2 for owner in issued))


def oracle_ceiling(people, owned, supports, cap):
    proof_masks = defaultdict(int)
    for i, owner in enumerate(people):
        known_docs = supports.get(owner, ())
        for doc, _ in owned[owner][:DEPTH]:
            if doc in known_docs:
                proof_masks[doc] |= 1 << i
    masks = set(proof_masks.values())
    undominated = [a for a in masks if not any(a != b and a & b == a for b in masks)]
    if not undominated or not cap:
        return 0
    return min(5, max(reduce(or_, group).bit_count()
                      for group in combinations(undominated, min(cap, len(undominated)))))


def run(qrels, output, reverse):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run in a Slurm allocation with a fresh path")
    experts, documents = load(qrels)
    text = read_topics(qrels / "ent06.expert.topics")
    topics = sorted(experts, key=int)
    index = HistoricalIndex(documents, topics, lexical_vectors(topics, text))
    histories = {}
    equal_reads = {}
    depth = defaultdict(Counter)
    for topic in reversed(topics) if reverse else topics:
        results = {}
        nominees = nominations(index, topic)
        supports = {owner: {doc for doc, label in docs.items() if label == 2}
                    for owner, docs in documents[topic].items()
                    if experts[topic].get(owner) == 2 and 2 in docs.values()}
        for arm in ARMS:
            people, owned = nominees[arm]
            people = people[:PERSON_CAP]
            for cutoff in (1, 4, DEPTH):
                depth[arm][cutoff] += sum(any(doc in supports.get(owner, ()) for doc, _ in owned[owner][:cutoff])
                                          for owner in people)
            results[arm] = {
                cap: {policy: allocate(people, owned, documents[topic], experts[topic], cap, policy)
                      for policy in POLICIES}
                for cap in READ_CAPS
            }
        histories[topic] = results
        ref = results[ARMS[0]][3]["rank_first"]
        if ref["checks"] != PAIR_CAP:
            raise ValueError("Realized reference check count changed")
        cap = ref["reads"]
        equal_reads[topic] = {}
        for arm in ARMS:
            people, owned = nominees[arm]
            people = people[:PERSON_CAP]
            runs = {mode: allocate(people, owned, documents[topic], experts[topic], cap, mode)
                    for mode in POLICIES}
            ceiling = oracle_ceiling(people, owned, supports, cap)
            if any(row["supported"] > ceiling or row["reads"] > cap for row in runs.values()):
                raise ValueError("Measured result exceeds its disclosed ceiling or read budget")
            equal_reads[topic][arm] = dict(oracle=ceiling, policies=runs)
        if equal_reads[topic][ARMS[0]]["policies"]["rank_first"] != ref:
            raise ValueError("Reference changed when its budget was replayed")
    summary = {}
    for arm in ARMS:
        summary[arm] = {"known_positive_with_direct_source": dict(depth[arm]), "read_caps": {}}
        for cap in READ_CAPS:
            output_policies = {}
            for policy in POLICIES:
                cases = [histories[t][arm][cap][policy] for t in topics]
                prior = [histories[t][ARMS[0]][cap][POLICIES[0]] for t in topics]
                if any(row["checks"] > PAIR_CAP or row["reads"] > cap for row in cases):
                    raise AssertionError("Work exceeds either cap")
                output_policies[policy] = dict(
                    **{key: sum(row[key] for row in cases) for key in cases[0]},
                    paired_supported_pp5_delta=paired_interval([
                        20 * (after["supported"] - before["supported"])
                        for after,before in zip(cases, prior, strict=True)]))
            summary[arm]["read_caps"][cap] = output_policies
    matched = {}
    reference = [histories[t][ARMS[0]][3]["rank_first"] for t in topics]
    for arm in ARMS:
        matched[arm] = dict(oracle_supported_slots=sum(equal_reads[t][arm]["oracle"] for t in topics))
        for policy in POLICIES:
            runs = [equal_reads[t][arm]["policies"][policy] for t in topics]
            matched[arm][policy] = dict(
                **{key: sum(row[key] for row in runs) for key in runs[0]},
                paired_supported_pp5_delta=paired_interval([
                    20 * (after["supported"] - before["supported"])
                    for after,before in zip(runs, reference, strict=True)]))
    write_json(output, dict(arms=summary, realized_reference_read_cap=matched, pair_cap=PAIR_CAP,
        owner_source_depth=DEPTH, topic_count=len(topics), reverse=reverse,
        comparison_signature=sha256(json.dumps([histories, equal_reads], sort_keys=True).encode()).hexdigest(),
        inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256},
        nomination_runner_sha256=digest(Path(__file__).with_name("nomination_waterfall.py")),
        runner_sha256=digest(Path(__file__)), slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("qrels", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
