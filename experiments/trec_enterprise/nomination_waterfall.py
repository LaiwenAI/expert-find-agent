"""Compare historical document-first and expert-first nomination on held topics."""

import argparse
from collections import Counter, defaultdict
from functools import cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics


PERSON_CAP = 20
DOCUMENT_CAP = 256
BUDGETS = (5, 13)
ARMS = ("document_first", "expert_first", "rank_fusion")


def lexical_vectors(topics, text):
    counts = [Counter(re.findall(r"[a-z0-9]+", text[topic].lower())) for topic in topics]
    frequency = Counter(word for row in counts for word in row)
    vocabulary = sorted(frequency)
    matrix = np.array([[(1 + math.log(row[word])) * math.log(1 + len(topics) / frequency[word])
                        if row[word] else 0 for word in vocabulary] for row in counts], dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def expert_votes(index, held):
    eligible = [i for i in range(len(index.topics)) if i != held]
    neighbor = sorted(eligible, key=lambda i: (-index.similarity[held, i], i))
    weights = {i: 1 / rank for rank, i in enumerate(neighbor, 1)}

    @cache
    def vote(mask):
        return sum(weights[i] for i in weights if mask & (1 << i))

    person_topics = {}
    sources = defaultdict(list)
    for doc, people in index.support.items():
        for owner, mask in people.items():
            observed = mask & ~(1 << held)
            if not observed:
                continue
            person_topics[owner] = person_topics.get(owner, 0) | observed
            sources[owner].append((doc, vote(observed)))
    votes = {owner: vote(mask) for owner, mask in person_topics.items()}
    ranked_sources = {owner: sorted(items, key=lambda row: (-row[1], row[0])) for owner, items in sources.items()}
    return sorted(votes, key=lambda owner: (-votes[owner], owner)), ranked_sources


def nominations(index, topic):
    held = index.positions[topic]
    doc_ranks = index.ranking(topic, {held})
    if len(doc_ranks) > DOCUMENT_CAP:
        raise ValueError("Changed historical document budget")
    document_sources = defaultdict(list)
    for doc, rank in doc_ranks.items():
        for owner, mask in index.support.get(doc, {}).items():
            retained = mask & ~(1 << held)
            if retained:
                count = retained.bit_count()
                score = ((count + .25) / (count + 1)) / math.sqrt(rank)
                document_sources[owner].append((doc, score))
    document_sources = {owner: sorted(items, key=lambda row: (-row[1], row[0]))
                        for owner, items in document_sources.items()}
    scores = {owner: sum(score for _, score in items[:3]) for owner, items in document_sources.items()}
    doc_order = sorted(scores, key=lambda owner: (-scores[owner], owner))
    person_order, person_sources = expert_votes(index, held)
    fusion = Counter()
    for order in (doc_order, person_order):
        for rank, owner in enumerate(order[:PERSON_CAP], 1):
            fusion[owner] += 1 / (10 + rank)
    fused_order = sorted(fusion, key=lambda owner: (-fusion[owner], owner))
    return {"document_first": (doc_order, document_sources),
            "expert_first": (person_order, person_sources),
            "rank_fusion": (fused_order, person_sources)}


def waterfall(known, sources, shortlist):
    reachable = {owner for owner, docs in known.items()
                 if any(doc in docs for doc, _ in sources.get(owner, ()))}
    shortlisted = reachable.intersection(shortlist)
    return dict(
        evidence_scope=len(reachable),
        shortlist_20=len(shortlisted),
        shortlist_5=len(shortlisted.intersection(shortlist[:5])),
        known_expert_in_20=len(set(known).intersection(shortlist)),
        sources_in_shortlist=sum(len(sources[owner]) for owner in shortlist),
    )


def replay(order, sources, expertise, evidence):
    checked = set()
    confirmed = {}
    trace = Counter()
    actions = [(owner, items[0][0]) for owner in order if (items := sources[owner])]
    for cap in BUDGETS:
        while len(checked) < cap:
            available = next((pair for pair in actions if pair not in checked and pair[0] not in confirmed), None)
            if available is None:
                extras = ((owner, doc) for owner in order if owner not in confirmed
                          for doc, _ in sources[owner] if (owner, doc) not in checked)
                available = next(extras, None)
            if available is None:
                break
            checked.add(available)
            owner, doc = available
            mark = evidence.get(owner, {}).get(doc)
            trace["unknown_pair_checks"] += mark is None
            if mark == 2:
                confirmed[owner] = doc
        selected = [owner for owner in order if owner in confirmed][:5]
        yield str(cap), dict(checked=len(checked), distinct_reads=len({doc for _, doc in checked}),
            issued=len(selected),
            supported_slots=sum(expertise.get(owner) == 2 for owner in selected), **trace)


def run(qrels, output, reverse):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require an allocated Slurm job and an unused output path")
    experts, documents = load(qrels)
    topics = sorted(experts, key=int)
    text = read_topics(qrels / "ent06.expert.topics")
    vectors = lexical_vectors(topics, text)
    index = HistoricalIndex(documents, topics, vectors)
    per_arm = {arm: {} for arm in ARMS}
    truth = Counter()
    query_order = list(reversed(topics)) if reverse else topics
    for topic in query_order:
        eligible = nominations(index, topic)
        # Rebuild without the held rows to verify independence of nominations and source ranks.
        no_held = HistoricalIndex({key: value for key,value in documents.items() if key != topic}, topics, vectors)
        if nominations(no_held, topic) != eligible:
            raise ValueError("Discovery depended on held-topic membership")
        known = {owner: {key for key, mark in docs.items() if mark == 2}
                 for owner, docs in documents[topic].items()
                 if experts[topic][owner] == 2 and 2 in docs.values()}
        truth.update(
            known_grade2_experts=sum(grade == 2 for grade in experts[topic].values()),
            known_grade2_experts_with_grade2_source=len(known),
            historical_grade2_pair_reachable=sum(any(
                index.support.get(doc, {}).get(owner, 0) & ~(1 << index.positions[topic])
                for doc in sources) for owner,sources in known.items()),
        )
        for arm in ARMS:
            ranking, sources = eligible[arm]
            order = ranking[:PERSON_CAP]
            measured = waterfall(known, sources, order)
            per_arm[arm][topic] = dict(waterfall=measured,
                                       budgets=dict(replay(order, sources, experts[topic], documents[topic])))
    output_arms = {}
    for arm in ARMS:
        results = per_arm[arm]
        per_budget = {}
        for cap in map(str, BUDGETS):
            observed = [results[q]["budgets"][cap] for q in topics]
            reference = [per_arm[ARMS[0]][q]["budgets"][cap] for q in topics]
            totals = sum((Counter(row) for row in observed), Counter())
            if totals["checked"] != int(cap) * len(topics) or any(
                    own["checked"] != base["checked"] for own,base in zip(observed,reference,strict=True)):
                raise ValueError("The paired realized budget differs")
            per_budget[cap] = dict(totals, paired_supported_pp5_delta=paired_interval([
                100*(after["supported_slots"]-before["supported_slots"])/5
                for after,before in zip(observed,reference,strict=True)]))
        output_arms[arm] = dict(
            waterfall=dict(sum((Counter(row["waterfall"]) for row in results.values()),Counter())),
            per_budget=per_budget,
        )
    signature = sha256(json.dumps(per_arm, sort_keys=True).encode()).hexdigest()
    write_json(output, dict(arms=output_arms, inventory=dict(truth), topic_count=len(topics),
        evidence_mode="deterministic held-out human judgment replay with unknown pairs withheld",
        comparison_signature=signature, reverse=reverse, inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256},
        runner_sha256=digest(Path(__file__)), slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("qrels", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
