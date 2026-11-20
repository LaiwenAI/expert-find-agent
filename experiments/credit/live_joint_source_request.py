"""Measure a local request path for cross-publication expert search."""

import argparse
from collections import Counter
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import elife_cross_publication_transfer as elife
from experiments.credit import joint_byline_depth as joint
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


BASE = joint.base
ARMS = ("topical", "parallel_byline", "owner_gap", "joint")
DEPTH = 64
TOPIC_TASK = "Decide whether this article's title and abstract concern the requested research topic."


def request(atoms):
    return json.dumps({"requirements": [{"topic": topic, "role": role} for topic, role in atoms]})


class LocalIndex:
    def __init__(self, catalogue, encoder, directory):
        self.catalogue = catalogue
        self.encoder = encoder
        self.ids = catalogue.ids
        self.valid = [i for i, doi in enumerate(self.ids) if catalogue.kinds[doi] is None]
        self.texts = {doi: "\n".join(catalogue.documents[doi]) for doi in self.ids}
        self.embeddings = encoder.encode([self.texts[doi] for doi in self.ids])
        directory.mkdir()
        self.paths = {}
        for i, doi in enumerate(self.ids):
            path = directory / f"{i}.json"
            path.write_text(json.dumps(catalogue.ownership[doi]["authors"], separators=(",", ":")))
            self.paths[doi] = path

    def rank(self, topics):
        dense = self.encoder.encode(topics, query=True) @ self.embeddings.T
        valid = set(self.valid)
        ranks = []
        for i, topic in enumerate(topics):
            lexical = self.catalogue.lexical.document_scores(topic)
            sparse = sorted((int(key) for key in lexical if int(key) in valid),
                            key=lambda key: (-lexical[str(key)], key))
            semantic = sorted(self.valid, key=lambda key: (-dense[i, key], key))
            fusion = Counter()
            for order in (sparse, semantic):
                for rank, key in enumerate(order, 1):
                    fusion[key] += 1 / (60 + rank)
            sequence = sorted(fusion, key=lambda key: (-fusion[key], key))[:DEPTH]
            ranks.append({self.ids[key]: j for j, key in enumerate(sequence, 1)})
        return tuple(ranks)

    def open_sources(self, documents):
        return {doi: {"authors": json.loads(self.paths[doi].read_bytes())} for doi in documents}


def simple_order(ranks):
    initial = list(dict.fromkeys(next(iter(facet)) for facet in ranks))
    if len(initial) == 1:
        initial.append(next(doi for doi in ranks[0] if doi != initial[0]))
    extras = sorted(set().union(*ranks) - set(initial), key=lambda doi: (
        min(facet.get(doi, DEPTH + 1) for facet in ranks),
        sum(facet.get(doi, DEPTH + 1) for facet in ranks), doi,
    ))
    if not extras:
        raise ValueError("Unexpected empty follow-up pool")
    return initial, extras


def inspect(index, atoms, ranks, arm):
    initial, extras = simple_order(ranks)
    byline = index.catalogue.byline
    if arm == "joint":
        plan = joint.posting_plan(byline, set().union(*ranks), ranks, [*initial, extras[0]], DEPTH)
        return plan, index.open_sources(plan)
    if arm == "parallel_byline":
        def gain(doi):
            return sum(len(byline[doi] & byline[prior])
                       for prior in initial for i in range(2)
                       if prior in ranks[i] and doi in ranks[1-i])
        extra = max(extras, key=gain)
    elif arm == "owner_gap":
        observed = index.open_sources(initial)
        partial = BASE.read_articles(SimpleNamespace(ownership=observed), initial, atoms, ranks)

        def gain(doi):
            return sum(1 / min(ranks[i][item] for item in seen)
                       for person, sides in partial.items() for i, seen in enumerate(sides)
                       if person in byline[doi] and seen and doi in ranks[1-i]
                       and not any(a != b for a, b in product(*sides)))
        extra = max(extras, key=gain)
        return [*initial, extra], observed | index.open_sources([extra])
    else:
        extra = extras[0]
    plan = [*initial, extra]
    return plan, index.open_sources(plan)


def render(loaded, plan, atoms, ranks):
    options = BASE.choose_proofs(BASE.read_articles(SimpleNamespace(ownership=loaded), plan, atoms, ranks), ranks)
    return json.dumps({"matches": options[:1]}, separators=(",", ":"))


def apply_topic_reader(reader, index, plan, atoms, ranks):
    scores, clipped = reader.score_pairs(
        [(topic, index.texts[doi]) for topic, _ in atoms for doi in plan], task=TOPIC_TASK,
    )
    if len(scores) != 2 * len(plan) or clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid topic verifier output")
    return tuple(
        {doi: facet[doi] for doi, margin in zip(plan, scores[i*len(plan):(i+1)*len(plan)], strict=True)
         if doi in facet and margin > 0.0}
        for i, facet in enumerate(ranks)
    )


def handle(index, reader, payload, arm, mode):
    start = perf_counter()
    atoms = tuple((item["topic"], item["role"]) for item in json.loads(payload)["requirements"])
    ranks = index.rank([topic for topic, _ in atoms])
    plan, loaded = inspect(index, atoms, ranks, arm)
    if len(loaded) != 3:
        raise AssertionError("The live policies must read exactly three declarations")
    if mode == "with_topic_reader":
        ranks = apply_topic_reader(reader, index, plan, atoms, ranks)
    result = render(loaded, plan, atoms, ranks)
    return result, (perf_counter() - start) * 1000


def experiment(index, reader, subjects, queries, offset):
    variants = [(mode, arm) for mode in ("structured", "with_topic_reader") for arm in ARMS]
    for mode, arm in variants:
        handle(index, reader, request(queries[0]), arm, mode)
    counts = {key: Counter() for key in variants}
    times = {key: [] for key in variants}
    local_known = {key: [] for key in variants}
    clusters = []
    for j, atoms in enumerate(queries):
        payload = request(atoms)
        clusters.append("|".join((atoms[0][0], atoms[1][0])))
        shift = (j + offset) % len(variants)
        for key in variants[shift:] + variants[:shift]:
            mode, arm = key
            result, ms = handle(index, reader, payload, arm, mode)
            matches = json.loads(result)["matches"]
            issued = bool(matches)
            known = issued and all(atoms[i][0] in subjects[matches[0][i+1]] for i in range(2))
            counts[key].update(issued=int(issued), known=int(known), tag_unresolved=int(issued and not known))
            times[key].append(ms)
            local_known[key].append(int(known))
    return {
        mode: {arm: dict(
            **counts[mode, arm], reads=3 * len(queries),
            p50_ms=float(np.quantile(times[mode, arm], .50)),
            p95_ms=float(np.quantile(times[mode, arm], .95)),
            paired_known_delta_vs_topical=clustered_paired_interval(
                np.asarray(local_known[mode, arm]) - local_known[mode, "topical"], clusters,
            ),
        ) for arm in ARMS}
        for mode in ("structured", "with_topic_reader")
    }


def verify_embedding(checkpoint):
    receipt = json.loads((checkpoint / "checkpoint-provenance.json").read_bytes())
    for name, expected in receipt["files"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Changed embedding checkpoint: {name}")
    return receipt["revision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result on a Slurm GPU")
    checkpoint = verify_embedding(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    plos, _ = BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, checksum = elife.load_catalogue(args.elife_xml, args.manifest)
    plos_subjects = BASE.publisher_topics([args.plos_xml])
    plos_gold = BASE.annotation_postings(plos, plos_subjects)
    life_subjects, _, life_queries = elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, BASE.query_panel(plos_gold)),
                  elife=(life, life_subjects, life_queries))
    with TemporaryDirectory() as scratch:
        outcome = {}
        for panel, (catalogue, subjects, queries) in panels.items():
            index = LocalIndex(catalogue, encoder, Path(scratch) / panel)
            outcome[panel] = dict(queries=len(queries), arms=experiment(index, reader, subjects, queries, args.order_offset))
            print(f"Completed {panel}: {len(queries)} requests", flush=True)
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), query_depth=DEPTH, order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), plos_manifest_sha256=digest(args.fixture / "manifest.json"),
        elife_manifest_sha256=digest(args.manifest), elife_projection_sha256=checksum,
        embedding_checkpoint=checkpoint, reranker_checkpoint=reader.provenance["revision"], panels=outcome,
    ))


if __name__ == "__main__":
    main()
