"""Certify a static expert response before opening source declarations."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from experiments.credit import linked_role_lane_index as linked
from experiments.credit.owner_evidence_bound import proof_key


ARMS = ("full", "two_only", "post_read", "pre_read")


def best_bound(plan, ranks, facets, optimistic):
    """Find the best feasible proof over the fixed plan and a role view."""
    lanes = [
        {doc: known | unknown if optimistic else known
         for doc in plan if doc in lane for known, unknown in (lane[doc],)}
        for lane in facets
    ]
    best = None
    for left, left_people in lanes[0].items():
        for right, right_people in lanes[1].items():
            if left == right:
                continue
            people = left_people & right_people
            if people:
                key = proof_key((min(people), left, right), ranks)
                if best is None or key < best:
                    best = key
    return None if best is None else best[-3:]


def verify(loaded, atoms, ranks):
    people = linked.replay.live.BASE.read_articles(
        SimpleNamespace(ownership=loaded), loaded, atoms, ranks)
    proofs = linked.replay.live.BASE.choose_proofs(people, ranks)
    return (proofs[0] if proofs else None), people


def execute(index, plan, ranks, facets, atoms, arm):
    if arm == "full":
        match, _ = verify(index.open_sources(plan), atoms, ranks)
        return match, len(plan), 1, False

    if arm == "pre_read":
        upper = best_bound(plan, ranks, facets, True)
        if upper is None:
            return None, 0, 0, False
        lower = best_bound(plan, ranks, facets, False)
        if upper != lower:
            match, _ = verify(index.open_sources(plan), atoms, ranks)
            return match, len(plan), 1, False
        read = list(upper[1:])
        loaded = index.open_sources(read)
        match, _ = verify(loaded, atoms, ranks)
        if match == upper:
            return match, len(read), 1, False
        missing = [doc for doc in plan if doc not in loaded]
        match, _ = verify(loaded | index.open_sources(missing), atoms, ranks)
        return match, len(plan), 2, True

    first = plan[:2]
    loaded = index.open_sources(first)
    match, people = verify(loaded, atoms, ranks)
    if arm == "two_only":
        return match, len(first), 1, False
    verified_facets = []
    for side, lane in enumerate(facets):
        verified = dict(lane)
        for doc in first:
            if doc in ranks[side]:
                verified[doc] = ({person for person, documents in people.items() if doc in documents[side]}, set())
        verified_facets.append(verified)
    if best_bound(plan, ranks, verified_facets, True) == match:
        return match, len(first), 1, False
    match, _ = verify(loaded | index.open_sources(plan[2:]), atoms, ranks)
    return match, len(plan), 2, False


def run_panel(catalogue, depth, embedding, scratch, reverse):
    index = linked.replay.live.LocalIndex(catalogue, linked.previous.mixed.QwenTorchEmbedding(embedding), scratch)
    indexes, _ = linked.certificate.indexes(catalogue)
    packed = linked.PackedLinks(indexes["full"])
    tags = {doc: {" ".join(topic.casefold().split()) for topic in row["topics"]}
            for doc, row in catalogue.ownership.items()}
    requests = linked.previous.live.BASE.query_panel(linked.previous.live.BASE.annotation_postings(catalogue, tags))
    # Stable withholding by source ID; still allow every repeated byline identity.
    hidden = {doc for doc in catalogue.ids if int(sha256(doc.encode()).hexdigest()[:8], 16) % 2}
    outcomes = {visibility: {arm: Counter() for arm in ARMS} for visibility in (100, 50)}
    timing = {(visibility, arm): [] for visibility in outcomes for arm in ARMS}
    signatures = {}
    order = list(enumerate(requests))
    if reverse:
        order.reverse()
    order.insert(0, (-1, requests[0]))
    for number, atoms in order:
        common_start = perf_counter()
        ranks = index.rank([topic for topic, _ in atoms], depth=depth)
        anchors, extras = linked.replay.live.simple_order(ranks, depth=depth)
        complete = tuple(packed.facet(lane, role) for lane, (_, role) in zip(ranks, atoms, strict=True))
        shared_seconds = perf_counter() - common_start
        for visibility in outcomes:
            plan_start = perf_counter()
            facets = complete if visibility == 100 else tuple(
                {doc: (set(), packed.compiled.eligible[doc]) if doc in hidden else view
                 for doc, view in lane.items()} for lane in complete)
            plan = linked.replay.typed.source_plan(ranks, facets, [*anchors, extras[0]], depth)
            if len(set(plan)) != len(plan) or len(plan) != 3:
                raise ValueError("Expected a three-source reference plan")
            common_ms = 1000 * (shared_seconds + perf_counter() - plan_start)
            base_arms = ARMS[::-1] if reverse else ARMS
            shift = number % len(ARMS)
            results = {}
            for arm in base_arms[shift:] + base_arms[:shift]:
                start = perf_counter()
                match, reads, waves, recomputed = execute(index, plan, ranks, facets, atoms, arm)
                result = json.dumps(match, separators=(",", ":"))
                milliseconds = 1000 * (perf_counter() - start)
                results[arm] = (result, reads, waves, recomputed)
                if number >= 0:
                    outcomes[visibility][arm].update(
                        reads=reads, waves=waves, recomputed=int(recomputed),
                        zero_reads=int(not reads), issued=int(match is not None),
                        tagged=int(match is not None and all(
                            topic in tags[doc] for (topic, _), doc in zip(atoms, match[1:], strict=True))),
                    )
                    timing[visibility, arm].append((common_ms + milliseconds, milliseconds))
            for arm in ("post_read", "pre_read"):
                if results[arm][0] != results["full"][0]:
                    raise ValueError(f"Exact route changed the issued person, citations or abstention: {arm}")
            if number >= 0:
                signatures[number, visibility] = (plan, results)
                outcomes[visibility]["two_only"]["changed"] += results["two_only"][0] != results["full"][0]
    return dict(
        requests=len(requests), sources=len(catalogue.ids), depth=depth,
        panels={visibility: {arm: dict(outcomes[visibility][arm],
                                       p95_local_ms=float(np.quantile(timing[visibility, arm], .95, axis=0)[0]),
                                       p95_dispatch_ms=float(np.quantile(timing[visibility, arm], .95, axis=0)[1]))
                             for arm in ARMS} for visibility in outcomes},
        decision_sha256=sha256(json.dumps(sorted(signatures.items()), sort_keys=True).encode()).hexdigest(),
    )


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU")
    catalogues = {name: linked.load(getattr(args, name), getattr(args, name + "_manifest"))
                  for name in linked.PANELS}
    left, right = catalogues.values()
    if (set(left.ids) & set(right.ids)
            or set().union(*left.byline.values()) & set().union(*right.byline.values())):
        raise ValueError("Holdout source and verified contributor sets must be disjoint")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = linked.previous.live.verify_embedding(args.embedding)
    if checkpoint != reference["embedding"]:
        raise ValueError("Changed embedding checkpoint")
    reports = {}
    with TemporaryDirectory(dir=args.output.parent) as directory:
        for name, catalogue in catalogues.items():
            if linked.prior.digest(getattr(args, name)) != reference["projection_sha256"][name]:
                raise ValueError("Snapshot differs from the frozen full-reader archive")
            for depth in linked.DEPTHS:
                panel = run_panel(catalogue, depth, args.embedding, Path(directory) / f"{name}-{depth}", args.reverse)
                expected = reference["panels"][name]["depths"][str(depth)]
                actual = panel["panels"][100]["full"]
                if any(actual[key] != expected[key] for key in ("issued", "tagged")):
                    raise ValueError("Full visibility baseline did not reproduce")
                reports[f"{name}_{depth}"] = panel
    linked.prior.write_json(args.output, dict(
        reports=reports, reference_sha256=linked.prior.digest(args.reference), embedding=checkpoint,
        script_sha256=linked.prior.digest(Path(__file__)), reverse=args.reverse,
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("extended", "extended_manifest", "confirmation", "confirmation_manifest", "embedding",
                  "reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
