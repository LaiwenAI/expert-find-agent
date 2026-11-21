"""Separate join planning from access to indexed contributor roles."""

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

from experiments.credit import mixed_publisher_joint as mixed
from experiments.followir.core17_reranking import digest, write_json

live, typed, audit = mixed.live, mixed.typed, mixed.audit
DEPTH = 128
ARMS = audit.ARMS  # owner_gap is the new same-role sequential control.


def role_sequential(index, atoms, ranks, facets):
    anchors = []
    for side, lane in enumerate(ranks):
        candidates = [doc for doc in lane if doc not in anchors]
        first_known = next((doc for doc in candidates if facets[side][doc][0]), None)
        anchors.append(first_known or candidates[0])
    observed = index.open_sources(anchors)
    by_person = live.BASE.read_articles(SimpleNamespace(ownership=observed), anchors, atoms, ranks)
    extras = sorted(set().union(*ranks) - set(anchors), key=lambda doc: (
        min(lane.get(doc, DEPTH + 1) for lane in ranks),
        sum(lane.get(doc, DEPTH + 1) for lane in ranks), doc,
    ))

    def gain(doc):
        strength = 0.
        for person, lanes in by_person.items():
            if any(a != b for a in lanes[0] for b in lanes[1]):
                continue
            for side, seen in enumerate(lanes):
                if seen and doc in facets[1-side] and person in facets[1-side][doc][0]:
                    strength += 1 / min(ranks[side][article] for article in seen)
        return strength

    third = max(extras, key=gain)
    return [*anchors, third], observed | index.open_sources([third])


def one(index, postings, atoms, arm):
    start = perf_counter()
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids))
    ranks = tuple(dict(list(lane.items())[:DEPTH]) for lane in full)
    facets = tuple(postings.facet(lane, label, 100, True)
                   for lane, (_, label) in zip(ranks, atoms, strict=True))
    if arm == "owner_gap":
        opened, loaded = role_sequential(index, atoms, ranks, facets)
    else:
        initial, extras = live.simple_order(ranks, depth=DEPTH)
        opened = typed.source_plan(ranks, facets, [*initial, extras[0]], DEPTH)
        loaded = index.open_sources(opened)
    if len(loaded) != 3 or set(opened) != loaded.keys():
        raise ValueError("Require exactly three distinct reads")
    proofs = live.BASE.choose_proofs(live.BASE.read_articles(
        SimpleNamespace(ownership=loaded), opened, atoms, ranks), ranks)
    payload = json.dumps(proofs[:1], separators=(",", ":"))
    return opened, proofs, ranks, payload, (perf_counter() - start) * 1000


def missing_bounds(rows, masks):
    # Identical query, person and evidence views share the unknown label.
    fixed = 0
    unknown = Counter()
    for arm, weight in (("owner_gap", -1), ("joint", 1)):
        for i in masks[arm]:
            row = rows[i]["arms"][arm]
            if row["known"]:
                fixed += weight
            else:
                unknown[(i, tuple(row["match"]))] += weight
    return [fixed + sum(min(0, value) for value in unknown.values()),
            fixed + sum(max(0, value) for value in unknown.values())]


def experiment(index, postings, queries, subjects, expected, reverse):
    tagged = lambda atoms, proof: all(topic in subjects.get(doc, ())
                                      for (topic, _), doc in zip(atoms, proof[1:], strict=True))
    rows = []
    latency = {arm: [] for arm in ARMS}
    reachable_after_reads = Counter()
    for arm in ARMS:
        one(index, postings, queries[0], arm)
    order = reversed(list(enumerate(queries))) if reverse else enumerate(queries)
    for number, atoms in order:
        row = dict(atoms=atoms, arms={}, topic_pair=json.dumps([topic for topic, _ in atoms]))
        arms = ARMS[::-1] if (number + reverse) % 2 else ARMS
        for arm in arms:
            opened, proofs, ranks, rendered, ms = one(index, postings, atoms, arm)
            match = next(iter(json.loads(rendered)), None)
            values = [ranks[i][doc] for i, doc in enumerate(match[1:])] if match else []
            row["arms"][arm] = dict(
                match=match, priority=(max(values), sum(values)) if values else None,
                known=bool(match) and tagged(atoms, match), opened=opened,
            )
            latency[arm].append(ms)
            reachable_after_reads[arm] += any(tagged(atoms, proof) for proof in proofs)
        rows.append((number, row))
    rows = [row for _, row in sorted(rows)]
    cases = {}
    for name, selected in audit.select(rows).items():
        cases[name] = audit.summarize(rows, selected) | {"joint_minus_sequential_missing_bounds":
                                                        missing_bounds(rows, selected)}
    reference = {field: expected[field] for field in ("known", "issued")}
    delta = {field: cases["natural"]["arms"]["joint"][field] - count for field, count in reference.items()}
    return dict(cases=cases, archived_joint_reference=reference, archived_joint_delta=delta,
                matched_reads_per_arm=3 * len(rows), reachable_after_reads=reachable_after_reads,
                p95_ms={key: float(np.quantile(value, .95)) for key, value in latency.items()},
                decision_signature=sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())
    provenance = json.loads(args.live_reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (projection != provenance["elife_projection_sha256"] or checkpoint != expected["embedding_checkpoint"]
            or digest(args.live_reference) != expected["live_reference_sha256"]
            or digest(args.fixture / "manifest.json") != provenance["plos_manifest_sha256"]
            or digest(args.elife_manifest) != provenance["elife_manifest_sha256"]
            or mixed.raw_identifiers(args.plos) & mixed.raw_identifiers(args.elife)):
        raise ValueError("Changed input or overlapping authors")
    plos_tags = base.publisher_topics([args.plos])
    elife_tags, _, elife_cases = live.elife.build_requests(elife)["topics"]
    cases = {
        "plos": (base.query_panel(base.annotation_postings(plos, plos_tags)), plos_tags),
        "elife": (elife_cases, elife_tags),
    }
    catalogue = mixed.merge(plos, elife)
    if len(catalogue.ids) != 452 or {key: len(panel[0]) for key, panel in cases.items()} != {"plos": 160, "elife": 51}:
        raise ValueError("Unexpected mixed catalog or query counts")
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = live.LocalIndex(catalogue, mixed.QwenTorchEmbedding(args.embedding), Path(scratch) / "reads")
        postings = typed.RolePostings(catalogue)
        panels = {key: experiment(index, postings, questions, tags,
                                  expected["panels"][key]["global128"]["arms"]["personal_roles"], args.reverse)
                  for key, (questions, tags) in cases.items()}
    write_json(args.output, dict(
        panels=panels, reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
        embedding=checkpoint, reverse=args.reverse, gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
