"""Measure fixed-budget open topic search with publisher-declared person roles."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit.legacy_contribution_text import stable
from experiments.credit.plos_role_inventory import declared_roles
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


FIXTURE = "74168d853ad21c795269c904fb7f8081cdf5a3e91ce7385ef8683495fd2ca7d5"
ARMS = ("topic_first", "role_union", "person_postings")
BUDGETS = (2, 4, 8)
CACHE = (50, 100)
RISK = .10
TASK = "Decide whether the publication substantively addresses the requested research topic."
TOP = 5


def compact(text):
    return " ".join(text.split())


def prepare(fixture, folders, output):
    if output.exists() or digest(fixture) != FIXTURE:
        raise ValueError("Expected a fresh output and the pinned role fixture")
    inputs = json.loads(fixture.read_bytes())["sources"]
    need = {row["source_sha256"]: row for row in inputs}
    captures = {}
    for folder in folders:
        for path in sorted(folder.glob("*.xml")):
            checksum = digest(path)
            if checksum in need:
                captures.setdefault(checksum, path)
    if need.keys() != captures.keys():
        raise ValueError("Missing pinned article snapshots")
    result = []
    excluded = []
    for record in sorted(inputs, key=lambda row: row["doi"]):
        checksum = record["source_sha256"]
        raw = captures[checksum].read_bytes()
        meta = ET.fromstring(raw).find("./front/article-meta")
        doi = record["doi"]
        if meta.findtext("article-id[@pub-id-type='doi']") != doi:
            raise ValueError("DOI does not match expected catalog")
        roles = declared_roles(raw, doi)
        if [tuple(sorted(row["roles"])) for row in record["authors"]] != list(map(tuple, roles)):
            raise ValueError("Publisher contributor declaration differs")
        rows = meta.findall("./article-categories/subj-group[@subj-group-type='Discipline-v3']")
        leaf = sorted({
            compact(g.findtext("subject", ""))
            for row in rows for g in row.iter("subj-group")
            if g.find("subj-group") is None and g.findtext("subject")
        })
        title_node = meta.find("./title-group/article-title")
        title = compact(" ".join(title_node.itertext())) if title_node is not None else ""
        abstract = meta.find("abstract")
        if not title or abstract is None:
            excluded.append(doi)
            continue
        result.append(dict(record, text=f"{title}\n{compact(' '.join(abstract.itertext()))}", topics=leaf))
    write_json(output, dict(fixture=FIXTURE, sources=result, excluded_missing_text=excluded))


class Catalog:
    def __init__(self, articles, encoder, reranker):
        self.articles = articles
        self.texts = [row["text"] for row in articles]
        self.encoder, self.reranker = encoder, reranker
        self.terms = [set(tokens(text)) for text in self.texts]
        frequency = Counter(token for terms in self.terms for token in terms)
        self.idf = {word: float(np.log1p((len(articles) - n + .5) / (n + .5)))
                    for word, n in frequency.items()}
        self.union, self.person, self.topics = defaultdict(set), defaultdict(set), defaultdict(set)
        self.bindings = {}
        for i, article in enumerate(articles):
            for topic in article["topics"]:
                self.topics[topic].add(i)
            role_union = set()
            for author in article["authors"]:
                owner = author["person"] or f"{article['doi']}#{author['ordinal']}"
                role_union.update(author["roles"])
                for pair in combinations(sorted(author["roles"]), 2):
                    self.person[pair].add(i)
                    self.bindings.setdefault((i, pair), []).append(owner)
            for pair in combinations(sorted(role_union), 2):
                self.union[pair].add(i)
        self.visible = {pct: {i for i, row in enumerate(articles)
                              if all(author["roles"] for author in row["authors"])
                              and int.from_bytes(stable("declaration-cache", row["doi"])[:8], "big") % 100 < pct}
                        for pct in CACHE}
        started = perf_counter()
        self.vectors = encoder.encode(self.texts)
        self.encode_seconds = perf_counter() - started

    def rank(self, topic, *, return_dense_scores=False):
        vector = self.encoder.encode([f"Research about {topic}"], query=True)[0]
        scores = self.vectors @ vector
        dense = np.argsort(-scores, kind="stable").tolist()
        words = set(tokens(topic))
        overlap = {i: sum(self.idf[w] for w in sorted(words & terms)) for i, terms in enumerate(self.terms)}
        lexical = sorted((i for i in dense if overlap[i]), key=lambda i: (-overlap[i], i))
        merged = {i: 1 / (60 + rank) for rank, i in enumerate(dense, 1)}
        for rank, i in enumerate(lexical, 1):
            merged[i] += 1 / (60 + rank)
        ranked = sorted(dense, key=lambda i: (-merged[i], i))
        return (ranked, scores) if return_dense_scores else ranked

    def candidates(self, ranked, pair, arm, visibility, limit):
        if arm == "topic_first":
            return ranked[:limit]
        posting = self.union[pair] if arm == "role_union" else self.person[pair]
        known = self.visible[visibility]
        selected = []
        for i in ranked:
            if i in posting or i not in known:
                selected.append(i)
                if len(selected) == limit:
                    break
        return selected

    def verify(self, topic, selected):
        if not selected:
            return []
        query = f"Research about {topic}"
        values, clipped = self.reranker.score_pairs([(query, self.texts[i]) for i in selected], task=TASK)
        if clipped or not np.isfinite(values).all():
            raise ValueError("Unexpected topical verifier output")
        return values


def query_panel(index, topics):
    for topic in topics:
        sources = index.topics[topic]
        support = {pair: len(sources & posts) for pair, posts in index.person.items()}
        eligible = [pair for pair, count in support.items() if 0 < count < len(sources)]
        if len(eligible) >= 2:
            rare = min(eligible, key=lambda pair: (support[pair], stable(topic, pair)))
            frequent = min((pair for pair in eligible if pair != rare),
                           key=lambda pair: (-support[pair], stable(topic, pair)))
            yield topic, "rare", rare
            yield topic, "frequent", frequent


def calibrate(index, topics):
    negative = []
    positive = []
    for topic in topics:
        selected = index.rank(topic)[:8]
        for i, value in zip(selected, index.verify(topic, selected), strict=True):
            (positive if i in index.topics[topic] else negative).append(value)
    negative.sort(reverse=True)
    cutoff = negative[int(RISK * len(negative))]
    return dict(cutoff=cutoff, off_label=len(negative), on_label=len(positive),
                accepted_off_label=sum(value > cutoff for value in negative),
                accepted_on_label=sum(value > cutoff for value in positive), topics=len(topics))


def recommendations(index, topic, pair, selected, scores, cutoff):
    present = index.topics[topic]
    evidence = {}
    actual = sorted(zip(selected, scores, strict=True), key=lambda row: -row[1])
    for article, score in actual:
        if score > cutoff:
            for owner in index.bindings.get((article, pair), ()):
                evidence.setdefault(owner, article)
                if len(evidence) == TOP:
                    break
        if len(evidence) == TOP:
            break
    return dict(issued=len(evidence), grounded=sum(i in present for i in evidence.values()),
                cited_off_label=sum(i not in present for i in evidence.values()))


def cross_budget_comparison(records):
    # Compare the observed four-versus-eight operating points on the same
    # requests. These descriptive intervals are not a noninferiority test.
    output = {}
    for coverage, budget in records.items():
        narrower = budget[4]["person_postings"]
        reference = budget[8]["topic_first"]
        if [(topic, group) for topic, group, _, _ in narrower] != [
                (topic, group) for topic, group, _, _ in reference]:
            raise ValueError("Unpaired operating points")
        groups = [topic for topic, _, _, _ in narrower]
        paired = list(zip(narrower, reference, strict=True))
        output[coverage] = {
            field: dict(clustered_paired_interval(
                [(a[2][field] - b[2][field]) / TOP for a, b in paired], groups),
                sampling_unit="whole topic")
            for field in ("grounded", "cited_off_label")
        }
    return output


def run(projection, embed, rerank, output):
    import torch
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise ValueError("Require fresh output and a CUDA compute allocation")
    prepared = json.loads(projection.read_bytes())
    articles = prepared["sources"]
    if prepared["fixture"] != FIXTURE or len(articles) + len(prepared["excluded_missing_text"]) != 152:
        raise ValueError("Unexpected source projection")
    provenance = verify_checkpoint(embed)
    encoder = QwenTorchEmbedding(embed)
    encoder.model.config.use_cache = False
    reranker = Reranker(rerank)
    catalog = Catalog(articles, encoder, reranker)
    eligible_topics = sorted(topic for topic, sources in catalog.topics.items() if len(sources) >= 4)
    division = sorted(eligible_topics, key=lambda topic: stable("topic-posting-split", topic))
    development, test = division[:len(division)//3], division[len(division)//3:]
    gate = calibrate(catalog, development)
    panels = list(query_panel(catalog, test))
    records = {percent: {budget: defaultdict(list) for budget in BUDGETS} for percent in CACHE}
    for order, (topic, difficulty, pair) in enumerate(panels):
        time = perf_counter()
        ranked = catalog.rank(topic)
        rank_ms = (perf_counter() - time) * 1000
        goal = catalog.topics[topic] & catalog.person[pair]
        if not goal:
            raise ValueError("Expected a declared topic-and-person association")
        offset = order % len(ARMS)
        for percent in CACHE:
            for budget in BUDGETS:
                for arm in ARMS[offset:] + ARMS[:offset]:
                    time = perf_counter()
                    selected = catalog.candidates(ranked, pair, arm, percent, budget)
                    scores = catalog.verify(topic, selected)
                    counts = recommendations(catalog, topic, pair, selected, scores, gate["cutoff"])
                    elapsed = rank_ms + (perf_counter() - time) * 1000
                    records[percent][budget][arm].append((
                        topic, difficulty, counts | dict(reads=len(selected), reachable=int(bool(goal & set(selected))),
                                                        scoped_sources_in_pool=len(goal & set(selected))),
                        elapsed,
                    ))
        if (order+1) % 20 == 0:
            print(f"Completed {order+1}/{len(panels)} topic-role queries", flush=True)
    result = {}
    for percent, budgets in records.items():
        result[percent] = {}
        for budget, arms in budgets.items():
            control = arms["topic_first"]
            report = {}
            for arm, items in arms.items():
                totals = sum((Counter(row) for _, _, row, _ in items), Counter())
                diffs = [(item[2]["grounded"] - ref[2]["grounded"]) / TOP
                         for item, ref in zip(items, control, strict=True)]
                report[arm] = dict(totals) | dict(
                    p95_ms=float(np.quantile([item[3] for item in items], .95)), queries=len(items),
                    grounded_p5_difference=dict(clustered_paired_interval(diffs, [item[0] for item in items]),
                                                 sampling_unit="whole topic"),
                    subsets={bucket: dict(sum((Counter(row) for _, label, row, _ in items if label == bucket),
                                            Counter())) for bucket in ("rare", "frequent")},
                )
            result[percent][budget] = report
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), fixture=FIXTURE, projection=digest(projection),
                            embedding=provenance["revision"], reranker=reranker.provenance["revision"],
                            sources=len(articles), excluded_missing_text=prepared["excluded_missing_text"],
                            sources_without_topics=sum(not row["topics"] for row in articles),
                            indexed={pct: len(rows) for pct, rows in catalog.visible.items()},
                            encode_seconds=catalog.encode_seconds, calibration=gate, test_topics=len(test),
                            test_queries=len(panels), results=result,
                            exploratory_person_four_vs_topic_eight=cross_budget_comparison(records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    for argument in ("fixture", "output"):
        prep.add_argument(argument, type=Path)
    prep.add_argument("folders", nargs="+", type=Path)
    replay = subparsers.add_parser("run")
    for argument in ("projection", "embed", "rerank", "output"):
        replay.add_argument(argument, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else run)(**args)
