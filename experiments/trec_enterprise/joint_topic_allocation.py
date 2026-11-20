"""Allocate three source checks across two independently judged TREC topics."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import os
from pathlib import Path
import re

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_byline_depth as join
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load


TOPIC_SHA256 = "103841c11f51744fb33595edfca193540322410425c60e3250f7060d837a4b60"
DEPTHS = (64, 256)
ARMS = ("topical", "sequential", "joint")


def read_topics(path):
    if digest(path) != TOPIC_SHA256:
        raise ValueError("Changed official TREC topic file")
    result = {}
    for section in re.findall(r"<top>(.*?)</top>", path.read_text(), flags=re.S):
        key = re.search(r"<num>\s*Number:\s*EX(\d+)", section).group(1)
        fields = [re.search(pattern, section, flags=re.S).group(1) for pattern in (
            r"<title>(.*?)</title>", r"<desc>\s*Description:\s*(.*?)</desc>",
        )]
        result[key] = "\n".join(" ".join(value.split()) for value in fields)
    return result


def evidence_labels(experts, documents):
    return {
        topic: {
            person: {doc for doc, grade in entries.items() if grade == 2}
            for person, entries in documents[topic].items()
            if labels[person] == 2 and 2 in entries.values()
        }
        for topic, labels in experts.items()
    }


class HistoricalIndex:
    def __init__(self, documents, topics, vectors):
        self.topics = tuple(topics)
        self.positions = {topic: i for i, topic in enumerate(topics)}
        owners = defaultdict(lambda: defaultdict(int))
        support = defaultdict(lambda: defaultdict(int))
        all_docs = {doc for panel in documents.values() for evidence in panel.values() for doc in evidence}
        self.ids = sorted(all_docs)
        self.id_positions = {doc: i for i, doc in enumerate(self.ids)}
        self.associations = np.zeros((len(topics), len(all_docs)), dtype=np.float32)
        for topic, candidates in documents.items():
            i = self.positions[topic]
            associated = set()
            for person, evidence in candidates.items():
                for doc, grade in evidence.items():
                    owners[doc][person] |= 1 << i
                    if grade == 2:
                        support[doc][person] |= 1 << i
                    associated.add(self.id_positions[doc])
            self.associations[i, list(associated)] = 1
        self.owners = dict(owners)
        self.support = dict(support)
        self.similarity = vectors @ vectors.T

    def ranking(self, topic, held):
        pos = self.positions[topic]
        remainder = sorted(
            (i for i in range(len(self.topics)) if i not in held),
            key=lambda i: (-self.similarity[pos, i], i),
        )
        weights = np.zeros(len(self.topics), dtype=np.float32)
        weights[remainder] = 1 / np.arange(1, len(remainder) + 1, dtype=np.float32)
        scores = weights @ self.associations
        valid = np.flatnonzero(scores > 0)
        order = valid[np.lexsort((valid, -scores[valid]))][:max(DEPTHS)]
        return {self.ids[position]: rank for rank, position in enumerate(order.tolist(), 1)}

    def contributor_sets(self, ranks, held, *, support_only=False):
        held_mask = sum(1 << i for i in held)
        associations = self.support if support_only else self.owners
        return {
            doc: {person for person, mask in associations.get(doc, {}).items() if mask & ~held_mask}
            for doc in set().union(*ranks)
        }


def observed(labels, people, ranks, files):
    matches = defaultdict(lambda: (set(), set()))
    for i, ranking in enumerate(ranks):
        for doc in files:
            if doc in ranking:
                for person in people[doc]:
                    if doc in labels[i].get(person, ()):
                        matches[person][i].add(doc)
    return matches


def proof_exists(known):
    return any(a and b and len(a | b) > 1 for a, b in known.values())


def source_plans(labels, candidates, ranks, depth):
    first, extra = live.simple_order(ranks, depth=depth)
    topical = [*first, extra[0]]
    ready = observed(labels, candidates, ranks, first)

    def gap_score(doc):
        return sum(
            1 / min(ranks[i][source] for source in supported)
            for person, sides in ready.items() for i, supported in enumerate(sides)
            if person in candidates[doc] and supported and doc in ranks[1-i]
            and not any(a != b for a, b in product(*sides))
        )

    repair = [*first, max(extra, key=gap_score)]
    together = join.posting_plan(candidates, set().union(*ranks), ranks, topical, depth)
    return dict(topical=topical, sequential=repair, joint=together)


def evaluate(index, experts, documents, offset):
    labels = evidence_labels(experts, documents)
    summaries = {
        view: {depth: dict(natural={arm: Counter() for arm in ARMS}, eligible=0,
                          available=0, disjoint={arm: Counter() for arm in ARMS}) for depth in DEPTHS}
        for view in ("all_pooled_links", "prior_grade_two_links")
    }
    held_pairs = [pair for pair in combinations(index.topics, 2) if any(
        len(labels[pair[0]][person] | labels[pair[1]][person]) > 1
        for person in labels[pair[0]].keys() & labels[pair[1]].keys()
    )]
    disjoint = set()
    used = set()
    for pair in sorted(held_pairs, key=lambda items: sha256("|".join(items).encode()).digest()):
        if used.isdisjoint(pair):
            disjoint.add(pair)
            used.update(pair)
    for j, pair in enumerate(held_pairs):
        held = {index.positions[t] for t in pair}
        full = tuple(index.ranking(topic, held) for topic in pair)
        query_gold = tuple(labels[topic] for topic in pair)
        for depth in DEPTHS:
            ranks = tuple(dict(list(side.items())[:depth]) for side in full)
            for view, summary in summaries.items():
                people = index.contributor_sets(ranks, held, support_only=view == "prior_grade_two_links")
                catalog = set().union(*ranks)
                any_proof = proof_exists(observed(query_gold, people, ranks, catalog))
                s = summary[depth]
                s["eligible"] += 1
                s["available"] += int(any_proof)
                plans = source_plans(query_gold, people, ranks, depth)
                shift = (j+offset) % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    files = plans[arm]
                    if len(files) != 3 or len(set(files)) != 3:
                        raise ValueError("Expected a three-source plan")
                    proof = proof_exists(observed(query_gold, people, ranks, files))
                    cross_checks = sum(len(people[doc]) for side in ranks for doc in files if doc in side)
                    record = dict(recovered=int(proof), document_checks=len(files), owner_links_exposed=cross_checks)
                    s["natural"][arm].update(**record)
                    if pair in disjoint:
                        s["disjoint"][arm].update(**record)
    return dict(topics=len(index.topics), catalog_articles=len(index.ids), positive_pairs=len(held_pairs),
                disjoint_pairs=len(disjoint), depths=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    checkpoint = live.verify_embedding(args.embedding)
    experts, documents = load(args.directory)
    text = read_topics(args.directory / "ent06.expert.topics")
    topics = sorted(experts, key=int)
    encoder = QwenTorchEmbedding(args.embedding)
    vectors = encoder.encode([text[topic] for topic in topics], query=True)
    index = HistoricalIndex(documents, topics, vectors)
    report = evaluate(index, experts, documents, args.order_offset)
    write_json(args.output, dict(
        **report, gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
        input_sha256={**FILES, "ent06.expert.topics": TOPIC_SHA256},
    ))


if __name__ == "__main__":
    main()
