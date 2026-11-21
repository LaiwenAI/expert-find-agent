"""Evaluate joint retrieval and same-owner proof with one query batch."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


AVAILABILITY = ("all", "left_missing", "right_missing")
POLICIES = ("dense_first", "hybrid_first", "linked", "supported", "anchored", "oracle_sources")
ROLE_VIEWS = ("oracle_roles", "ridge_top1", "ridge_band", "prototype_top1", "prototype_band", "top1_agreement")
KEYWORD_STOP = frozenset("the a an and from for of to in by with on it this that which as at or is are be was were".split())
TOPIC_TASK = "Find the engineering publication that reports these results."
DEPTH = 6
MAX_SCORE_GAP = .04


class Catalog:
    def __init__(self, articles, encoder):
        self.articles = articles
        self.ids = sorted(articles)
        passages = [articles[key]["indexed"] for key in self.ids]
        self.vectors = encoder.encode(passages)
        terms = [Counter(normalize(text).split()) for text in passages]
        freq = Counter(word for row in terms for word in row)
        self.idf = {word: math.log1p((len(terms) - count + .5) / (count + .5)) for word, count in freq.items()}
        length = sum(sum(row.values()) for row in terms) / len(terms)
        self.lexical = [{word: self.idf[word] * 2.2 * n / (n + .3 + .9 * sum(row.values()) / length)
                        for word, n in row.items()} for row in terms]

    def keywords(self, text):
        words = list(dict.fromkeys(normalize(text).split()))
        available = [word for word in words if word not in KEYWORD_STOP and len(word) > 2 and word in self.idf]
        return " ".join(sorted(available, key=lambda token: -self.idf[token])[:8])

    def rank(self, vectors, texts, withheld, *, depth=DEPTH):
        dense = vectors @ self.vectors.T
        kept = [i for i, doc in enumerate(self.ids) if doc != withheld]
        topic = []
        for scores, query in zip(dense, texts, strict=True):
            lex = set(normalize(query).split()) - KEYWORD_STOP
            sparse = [sum(terms.get(word, 0.) for word in sorted(lex)) for terms in self.lexical]
            d = sorted(kept, key=lambda i: (-scores[i], self.ids[i]))
            s = sorted(kept, key=lambda i: (-sparse[i], self.ids[i]))
            a = {j: i for i, j in enumerate(d)}
            b = {j: i for i, j in enumerate(s)}
            fusion = sorted(kept, key=lambda j: (-(1 / (61 + a[j]) + 1 / (61 + b[j])), self.ids[j]))
            topic.append(dict(dense=[self.ids[i] for i in d[:depth]], fused=[self.ids[i] for i in fusion[:depth]],
                              cosine={self.ids[i]: float(scores[i]) for i in kept}, best=float(scores[d[0]])))
        return topic


def phrase_bank(wording):
    table = defaultdict(list)
    for i, _, phrase in wording["niso"]:
        table[ROLES[i]].append(phrase)
    return table


def required_roles(query, role_scores, mode, bands):
    if mode == "oracle_roles":
        return [{role} for role in query["roles"]]
    if mode == "top1_agreement":
        return [{ROLES[int(a.argmax())], ROLES[int(b.argmax())]}
                for a, b in zip(role_scores["ridge"], role_scores["prototype"], strict=True)]
    bank, scheme = mode.split("_")
    if scheme == "top1":
        return [{ROLES[int(row.argmax())]} for row in role_scores[bank]]
    return [{role for i, role in enumerate(ROLES) if row.max() - row[i] <= bands[bank]}
            for row in role_scores[bank]]


def candidates(catalog, pair, required):
    if len(set(pair)) != 2:
        return []
    a, b = [catalog.articles[key]["roles"] for key in pair]
    return sorted(owner for owner in a.keys() & b.keys() if required[0].issubset(a[owner]) and required[1].issubset(b[owner]))


def select(catalog, topic, query, required, policy, withheld):
    if policy == "oracle_sources":
        if withheld in query["papers"]:
            return None, []
        pair = tuple(query["papers"])
        return pair, candidates(catalog, pair, required)
    if policy in ("dense_first", "hybrid_first"):
        kind = "dense" if policy == "dense_first" else "fused"
        pair = tuple(side[kind][0] for side in topic)
        return pair, candidates(catalog, pair, required)
    ordered = sorted(product(*(side["fused"] for side in topic)),
                     key=lambda pair: (sum(side["fused"].index(p) for side, p in zip(topic, pair, strict=True)), pair))
    for pair in ordered:
        if pair[0] == pair[1]:
            continue
        if policy == "anchored":
            if not any(side["fused"][0] == p for side, p in zip(topic, pair, strict=True)):
                continue
            if any(side["best"] - side["cosine"][p] > MAX_SCORE_GAP for side, p in zip(topic, pair, strict=True)):
                continue
        if policy == "linked":
            joined = candidates(catalog, pair, [set(), set()])
            if joined:
                return pair, candidates(catalog, pair, required)
        else:
            owners = candidates(catalog, pair, required)
            if owners:
                return pair, owners
    return None, []


def evaluate(data, encoder, banks, wording, bands, offset):
    articles = data["articles"]
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    report = defaultdict(Counter)
    latencies = defaultdict(list)
    trace = []
    cases = sorted(data["queries"], key=lambda row: (row["family"], row["variant"]), reverse=bool(offset))
    for iteration, query in enumerate([cases[0], *cases]):
        family = query["family"]
        descriptions = [phrases[r][int(family[:16], 16) % len(phrases[r])] for r in query["roles"]]
        abstract = [articles[paper]["clue"] for paper in query["papers"]]
        for wording_name, clues in (("excerpt", abstract), ("keywords", [catalog.keywords(x) for x in abstract])):
            start = perf_counter()
            prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            prompts.extend(f"Instruct: {TASK}\nQuery:{phrase}" for phrase in descriptions)
            v = encoder.encode(prompts)
            role_scores = {bank: v[2:] @ banks[bank].T for bank in bands}
            encode_ms = (perf_counter() - start) * 1000
            for status in AVAILABILITY:
                withheld = None if status == "all" else query["papers"][0 if status == "left_missing" else 1]
                start = perf_counter()
                topic = catalog.rank(v[:2], clues, withheld)
                topic_ms = (perf_counter() - start) * 1000
                for mode in ROLE_VIEWS:
                    start = perf_counter()
                    required = required_roles(query, role_scores, mode, bands)
                    role_ms = (perf_counter() - start) * 1000
                    for policy in POLICIES:
                        start = perf_counter()
                        pair, owners = select(catalog, topic, query, required, policy, withheld)
                        json.dumps([pair, owners])
                        elapsed = encode_ms + topic_ms + role_ms + (perf_counter() - start) * 1000
                        if not iteration:
                            continue
                        key = (wording_name, status, mode, policy)
                        sanctioned = query["gold"] if withheld is None and tuple(query["papers"]) == pair else []
                        exact = bool(sanctioned) and owners == sanctioned
                        selected_right = pair == tuple(query["papers"])
                        covered_roles = all(role in kept for role, kept in zip(query["roles"], required, strict=True))
                        counts = report[key]
                        counts.update(
                            requests=1, exact_positive=int(exact), issued=int(bool(owners)),
                            exact_positive_with_intended_roles=int(exact and covered_roles),
                            wrong_scope_issued=int(bool(owners) and not selected_right),
                            unsupported_same_scope=int(bool(set(owners) - set(sanctioned)) and selected_right),
                            exact_pair=int(selected_right), fully_nominated=int(all(
                                paper in side["fused"] for paper, side in zip(query["papers"], topic, strict=True))),
                            intended_roles_in_band=int(covered_roles),
                        )
                        latencies[key].append(elapsed)
                        trace.append(dict(family=family, variant=query["variant"], key=key,
                                          pair=pair, issued=len(owners), exact=exact))
    result = {"|".join(key): dict(counts) | {"p95_ms": float(np.quantile(latencies[key], .95))}
              for key, counts in sorted(report.items())}
    return result, sorted(trace,key=lambda row:(row["family"],row["variant"],row["key"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "manifest", "examples", "training", "calibration", "embedding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    if digest(args.panel) != json.loads(args.manifest.read_bytes())["panel_sha256"]:
        raise ValueError("Source panel differs from the manifest")
    calibration = json.loads(args.calibration.read_bytes())["runs"][0]["results"]["sensors"]["original"]["results"]
    bands = {bank: calibration[bank]["threshold"] for bank in ("prototype", "ridge")}
    definitions, phrases, wording = queries(args.examples, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, phrases)
    input_panel = json.loads(args.panel.read_bytes())
    outcomes, trace = evaluate(input_panel, encoder, banks, wording, bands, args.offset)
    write_json(args.output, dict(
        articles=len(input_panel["articles"]), requests=len(input_panel["queries"]), bands=bands,
        source_sha256=digest(args.panel), calibration_sha256=digest(args.calibration),
        script_sha256=digest(Path(__file__)), embedding_revision=checkpoint["revision"],
        gpu=torch.cuda.get_device_name(), offset=args.offset, results=outcomes, trace=trace,
    ))
