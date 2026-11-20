"""Skip deeper first-round planning when shallow source proofs certify the answer."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_prefix_completion as prefix
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import live_joint_source_request as live
from experiments.credit import mixed_publisher_joint as mixed
from experiments.credit import robust_role_postings as stress
from experiments.followir.core17_reranking import digest, write_json


FULL_DEPTH = 128
SHALLOW_DEPTHS = (32, 64)
SETTINGS = ((0, 0), (50, 0), (50, 20), (100, 0), (100, 20))


def truncate(ranks, cap):
    return tuple(dict(list(side.items())[:cap]) for side in ranks)


def deepest_plan(index, atoms, ranks, role_view, shallow_facets=None):
    if shallow_facets is None:
        facets = stress.facets_for(ranks, atoms, index.catalogue.byline, role_view)
    else:
        tails = tuple({doi: rank for doi, rank in side.items() if doi not in shallow}
                      for side, shallow in zip(ranks, shallow_facets, strict=True))
        rest = stress.facets_for(tails, atoms, index.catalogue.byline, role_view)
        facets = tuple(shallow | tail for shallow, tail in zip(shallow_facets, rest, strict=True))
    initial, extras = live.simple_order(ranks, depth=FULL_DEPTH)
    return typed.source_plan(ranks, facets, [*initial, extras[0]], FULL_DEPTH)


def top_proof(loaded, atoms, ranks):
    options = live.BASE.choose_proofs(prefix.evidence(loaded, atoms, ranks), ranks)
    return next(iter(options), None)


def fixed(index, atoms, ranks, role_view):
    plan = deepest_plan(index, atoms, ranks, role_view)
    loaded = index.open_sources(plan)
    if len(loaded) != 3:
        raise ValueError("Expected three distinct sources in the fixed plan")
    return top_proof(loaded, atoms, ranks)


def adaptive(index, atoms, ranks, role_view, depth):
    shallow = truncate(ranks, depth)
    byline = index.catalogue.byline
    facets = stress.facets_for(shallow, atoms, byline, role_view)
    first = typed.first_pair(shallow, facets)
    loaded = {}
    cause = "no_indexed_pair"
    if first is not None and not any(row[0] for row in first):
        anchors = [row[2] for row in first]
        loaded = index.open_sources(anchors)
        observed = prefix.evidence(loaded, atoms, shallow)
        checked = next(iter(live.BASE.choose_proofs(observed, shallow)), None)
        cause = "indexed_pair_unconfirmed"
        if checked is not None:
            cause = "possible_shallow_displacement"
            bound = prefix.alternative_proofs(observed, shallow, byline, anchors)
            if not bound or prefix.proof_key(checked, shallow) <= min(row[0] for row in bound):
                return checked, 2, sum(map(len, shallow)), "certified"

    plan = deepest_plan(index, atoms, ranks, role_view, facets)
    if loaded and list(loaded) != plan[:2]:
        raise AssertionError("An exact shallow pair did not remain the deep anchor pair")
    loaded.update(index.open_sources([doc for doc in plan if doc not in loaded]))
    if len(loaded) != 3:
        raise AssertionError("Adaptive fallback must read exactly the fixed plan's three sources")
    return top_proof(loaded, atoms, ranks), 3, sum(map(len, ranks)), cause


def evaluate(index, subjects, queries, offset):
    postings = typed.RolePostings(index.catalogue)
    work = [(atoms, index.rank([topic for topic, _ in atoms], depth=FULL_DEPTH)) for atoms in queries]
    result = {}
    for coverage, corruption in SETTINGS:
        masks = stress.MASKS if (coverage, corruption) in ((50, 0), (50, 20), (100, 20)) else ("",)
        result[f"{coverage}-{corruption}"] = []
        for salt in masks:
            view, inventory = stress.project(postings, index.catalogue, coverage, corruption, salt)
            summaries = {depth: Counter() for depth in SHALLOW_DEPTHS}
            fixed_summary = Counter()
            for number, (atoms, ranks) in enumerate(work):
                reference = fixed(index, atoms, ranks, view)
                known = bool(reference) and all(
                    topic in subjects.get(doi, ()) for (topic, _), doi in zip(atoms, reference[1:], strict=True)
                )
                fixed_summary.update(known=int(known), issued=int(reference is not None))
                depths = SHALLOW_DEPTHS if (number + offset) % 2 == 0 else SHALLOW_DEPTHS[::-1]
                for depth in depths:
                    actual, reads, inspected, cause = adaptive(index, atoms, ranks, view, depth)
                    if actual != reference:
                        raise AssertionError("Adaptive depth changed a fixed-depth expert or citation")
                    summaries[depth].update(equal=1, reads=reads, candidate_rows_projected=inspected)
                    summaries[depth][cause] += 1
            result[f"{coverage}-{corruption}"].append(
                dict(salt=salt, inventory=inventory, fixed=fixed_summary, adaptive=summaries)
            )
    return result


def live_timing(index, queries, offset):
    timings = {name: [] for name in ("fixed128", "adaptive32", "adaptive64")}
    postings = typed.RolePostings(index.catalogue)
    for coverage, corruption, salt in ((100, 0, ""), (50, 20, "")):
        planned, _ = stress.project(postings, index.catalogue, coverage, corruption, salt)
        times = {name: [] for name in timings}
        for j, pair in enumerate(queries):
            payload = live.request(pair)
            names = tuple(times)
            shift = (j + offset) % len(names)
            reference = {}
            for arm in names[shift:] + names[:shift]:
                start = perf_counter_ns()
                atoms = tuple((obj["topic"], obj["role"]) for obj in json.loads(payload)["requirements"])
                ranks = index.rank([topic for topic, _ in atoms], depth=FULL_DEPTH)
                if arm == "fixed128":
                    proof = fixed(index, atoms, ranks, planned)
                else:
                    depth = int(arm[len("adaptive"):])
                    proof, _, _, _ = adaptive(index, atoms, ranks, planned, depth)
                reference[arm] = json.dumps({"matches": [proof] if proof else []}, separators=(",", ":"))
                times[arm].append((perf_counter_ns() - start) / 1e6)
            if len(set(reference.values())) != 1:
                raise AssertionError("Live timing route changed response bytes")
        for name in timings:
            timings[name].append(dict(
                coverage=coverage, corruption=corruption,
                p95_ms=float(np.quantile(times[name], .95)),
            ))
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    elife, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"]
    ):
        raise ValueError("Frozen inputs differ from reference")
    if mixed.raw_identifiers(args.plos_xml) & mixed.raw_identifiers(args.elife_xml):
        raise ValueError("A mixed catalog with shared identities requires a linked fixture")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_cases = base.query_panel(base.annotation_postings(plos, plos_subjects))
    elife_subjects, _, elife_cases = live.elife.build_requests(elife)["topics"]
    panels = dict(plos=(plos_subjects, plos_cases), elife=(elife_subjects, elife_cases))
    catalogue = mixed.merge(plos, elife)
    model = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory() as scratch:
        index = live.LocalIndex(catalogue, model, Path(scratch) / "sources")
        results = {
            name: dict(queries=len(queries), scenarios=evaluate(index, subjects, queries, args.order_offset),
                       timing=live_timing(index, queries, args.order_offset))
            for name, (subjects, queries) in panels.items()
        }
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), order_offset=args.order_offset, embedding_checkpoint=checkpoint,
        articles=len(catalogue.ids), fixed_depth=FULL_DEPTH, shallow_depths=SHALLOW_DEPTHS,
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.reference), panels=results,
    ))


if __name__ == "__main__":
    main()
