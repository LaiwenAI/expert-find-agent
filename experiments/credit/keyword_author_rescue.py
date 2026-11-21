"""Recover a second paper through byline postings and independent topic checks."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_one_role_transfer import build as development
from experiments.credit.elife_prompt_holdout import build as holdout
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, KEYWORD_STOP, TOPIC_TASK, Catalog, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.role_issuance_audit import outcome
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


GRID = tuple(product((False, True), (.04, .08, .12), (.25, .50, .75)))
WIDTH = 24
UNGATED = (True, float("inf"), 0.)


class PostingCatalog(Catalog):
    def __init__(self, articles, encoder):
        super().__init__(articles, encoder)
        self.owner_papers = defaultdict(set)
        for paper, row in articles.items():
            for owner in row["roles"]:
                self.owner_papers[owner].add(paper)

    def full_rank(self, vectors, clues, withheld):
        dense = vectors @ self.vectors.T
        kept = [i for i, doc in enumerate(self.ids) if doc != withheld]
        result = []
        for cosine, clue in zip(dense, clues, strict=True):
            words = sorted(set(normalize(clue).split()) - KEYWORD_STOP)
            sparse = [sum(doc.get(word, 0.) for word in words) for doc in self.lexical]
            d = {i: pos for pos, i in enumerate(sorted(kept, key=lambda i: (-cosine[i], self.ids[i])))}
            s = {i: pos for pos, i in enumerate(sorted(kept, key=lambda i: (-sparse[i], self.ids[i])))}
            fused = sorted(kept, key=lambda i: (-(1 / (61 + d[i]) + 1 / (61 + s[i])), self.ids[i]))
            high = max(sparse[i] for i in kept)
            result.append(dict(
                fused=[self.ids[i] for i in fused[:WIDTH]],
                rank={self.ids[i]: pos for pos, i in enumerate(fused[:WIDTH])},
                cosine={self.ids[i]: float(cosine[i]) for i in kept}, best=float(max(cosine[i] for i in kept)),
                lexical={self.ids[i]: sparse[i] / high if high else 0. for i in kept},
            ))
        return result

    def linked_options(self, topic):
        options = []
        examined = 0
        for slot, side in enumerate(topic):
            anchor = side["fused"][0]
            rival = topic[1 - slot]
            anchor_gap = side["best"] - side["cosine"][anchor]
            anchor_lexical = side["lexical"][anchor]
            neighbors = set().union(*(self.owner_papers[owner] for owner in self.articles[anchor]["roles"]))
            pool = sorted((neighbors - {anchor}) & rival["rank"].keys())
            examined += len(pool)
            for partner in pool:
                gap = rival["best"] - rival["cosine"][partner]
                lexical = rival["lexical"][partner]
                pair = (anchor, partner) if slot == 0 else (partner, anchor)
                options.append((rival["rank"][partner], pair, anchor_gap, anchor_lexical, gap, lexical))
        return sorted(options), examined


def at_gate(options, symmetric, gap, lexical):
    for _, pair, anchor_gap, anchor_overlap, partner_gap, partner_overlap in options:
        anchor_ok = anchor_gap <= gap and anchor_overlap >= lexical if symmetric else anchor_gap <= .04
        if anchor_ok and partner_gap <= gap and partner_overlap >= lexical:
            return pair
    return None


def sample(articles, cases, encoder, banks, wording, bands, reverse):
    catalog = PostingCatalog(articles, encoder)
    phrases = phrase_bank(wording)
    rows = defaultdict(list)
    if reverse:
        cases = cases[::-1]
    for i, row in enumerate([cases[0], *cases]):
        target = tuple(row["papers"])
        excerpts = [articles[doc]["clue"] for doc in target]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(value) for value in excerpts])):
            start = perf_counter()
            vectors = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                     [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in text])
            scores = {"prototype": vectors[2:] @ banks["prototype"].T}
            needed = required_roles(row, scores, "prototype_band", bands)
            embed_ms = (perf_counter() - start) * 1000
            for status, masked in zip(AVAILABILITY, (None, *target), strict=True):
                start = perf_counter()
                topic = catalog.full_rank(vectors[:2], clues, masked)
                retrieval_ms = (perf_counter() - start) * 1000
                shallow = [side | {"fused": side["fused"][:6]} for side in topic]
                start = perf_counter()
                base, _ = select(catalog, shallow, {}, [set(), set()], "anchored", masked)
                base_ms = embed_ms + retrieval_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                deep, _ = select(catalog, topic, {}, [set(), set()], "anchored", masked)
                deep_ms = embed_ms + retrieval_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                options, examined = catalog.linked_options(topic) if base is None else ([], 0)
                postings_ms = (perf_counter() - start) * 1000

                def report(pair):
                    static, _ = outcome(catalog, row, pair, needed, "evidence", None)
                    oracle, _ = outcome(catalog, row, pair, [{role} for role in row["roles"]], "evidence", None)
                    return dict(pair_correct=pair == target, pair_issued=pair is not None, static=static, oracle=oracle)

                if i:
                    timed = dict(anchored6=(report(base), base_ms), anchored24=(report(deep), deep_ms))
                    for policy in (*GRID, UNGATED):
                        start = perf_counter()
                        pair = base or at_gate(options, *policy)
                        delay = base_ms + postings_ms + (perf_counter() - start) * 1000
                        timed[policy] = report(pair), delay
                    visible = both = within = one_leader = False
                    if masked is None and row["variant"] == "present":
                        visible = all(doc in side["fused"][:6] for doc, side in zip(target, topic, strict=True))
                        both = all(doc in side["fused"] for doc, side in zip(target, topic, strict=True))
                        within = all(side["best"] - side["cosine"][doc] <= .04
                                     for doc, side in zip(target, topic, strict=True))
                        one_leader = any(doc == side["fused"][0] for doc, side in zip(target, topic, strict=True))
                    rows[view, status].append(dict(
                        positive=row["variant"] == "present", both_top6=visible, both_top24=both,
                        both_cosine_gate=within, one_leader=one_leader, examined=examined, routes=timed))
    return rows


def summarize(rows, selected):
    results = {}
    for (view, status), requests in sorted(rows.items()):
        counts = Counter()
        for row in requests:
            counts.update({k:v for k,v in row.items() if k != "routes"})
        methods = {}
        for arm, label in (("anchored6", "anchored6"), ("anchored24", "anchored24"),
                           *selected[view].items(), ("ungated", UNGATED)):
            values = [row["routes"][label] for row in requests]
            pair, static, oracle = Counter(), Counter(), Counter()
            local = []
            for outcome, ms in values:
                pair.update(correct=outcome["pair_correct"], issued=outcome["pair_issued"])
                static.update(outcome["static"])
                oracle.update(outcome["oracle"])
                local.append(ms)
            methods[arm] = dict(pair=dict(pair), static=dict(static), oracle=dict(oracle),
                                p95_source_ms=float(np.quantile(local, .95)))
        results[f"{view}|{status}"] = dict(census=dict(counts), routes=methods)
    return results


def fit(development_rows):
    selected = {}
    for view in ("excerpt", "keywords"):
        valid = development_rows[view, "all"]
        misses = development_rows[view, "left_missing"] + development_rows[view, "right_missing"]
        limits = sum(row["routes"]["anchored6"][0]["oracle"]["substitute"] for row in misses)
        selected[view] = {}
        for symmetric, name in ((False, "postings"), (True, "symmetric_postings")):
            eligible = []
            for policy in GRID:
                mode, gap, lexical = policy
                if mode != symmetric:
                    continue
                substitutions = sum(row["routes"][policy][0]["oracle"]["substitute"] for row in misses)
                if substitutions > limits:
                    continue
                positives = sum(row["routes"][policy][0]["oracle"]["proof"] for row in valid)
                pair_hits = sum(row["routes"][policy][0]["pair_correct"] for row in valid)
                eligible.append((positives, pair_hits, -substitutions, -gap, lexical, policy))
            selected[view][name] = max(eligible)[-1] if eligible else "anchored6"
    return selected


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    source, queries_dev, dev = development(args.development)
    calibration = sample(source, queries_dev, encoder, banks, wording, bands, args.reverse)
    params = fit(calibration)
    articles, cases, panel = holdout(args.holdout, json.loads(args.manifest.read_bytes()))
    outcomes = sample(articles, cases, encoder, banks, wording, bands, args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), dev_sha256=digest(args.development), holdout_sha256=digest(args.holdout),
        embedding_revision=checkpoint["revision"], gpu=torch.cuda.get_device_name(), reverse=args.reverse,
        dev=dev, held=panel, selected=params, calibration=summarize(calibration, params), results=summarize(outcomes, params),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "holdout", "manifest", "niso", "training", "bands", "embedding", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
