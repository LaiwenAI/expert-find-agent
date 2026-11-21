"""Verify exact query-local pruning on two independent publisher snapshots."""

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

from experiments.credit import open_topic_link_certificate as certificate
from experiments.credit.role_pair_placement import DeclarationIndex

prior, replay, previous = certificate.prior, certificate.replay, certificate.previous
ARMS = ("full", "global_links", "lane_links", "packed_links", "packed_lane")
PANELS = ("extended", "confirmation")
DEPTHS = (32, 128)


class LaneLinks:
    """Keep sources that can join a distinct source in the opposite rank list."""

    def __init__(self, byline, linked_docs):
        self.bit = {doc: 1 << i for i, doc in enumerate(sorted(linked_docs))}
        owners = {}
        for doc, bit in self.bit.items():
            for person in byline[doc]:
                owners[person] = owners.get(person, 0) | bit
        self.neighbors = {}
        for doc, bit in self.bit.items():
            linked = 0
            for person in byline[doc]:
                linked |= owners[person]
            self.neighbors[doc] = linked ^ bit

    def filter(self, ranks):
        # Each worklist has unique source IDs, so addition is bit union.
        bits = tuple(sum(self.bit.get(doc, 0) for doc in lane) for lane in ranks)
        return tuple(
            {doc: rank for doc, rank in lane.items()
             if self.neighbors.get(doc, 0) & bits[1 - side]}
            for side, lane in enumerate(ranks)
        )

    def payload(self):
        # Stable linked ordinals are already shared. Each stored mask is an
        # independently length-prefixed little-endian integer.
        return sum(4 + (bits.bit_length() + 7) // 8 for bits in self.neighbors.values())


class PackedLinks:
    """Use the prior exact repeated-author projection on active sources."""

    def __init__(self, full):
        self.compiled = DeclarationIndex(full, bridge_only=True)
        sizes = self.compiled.costs
        self.indexed, self.payload = self.compiled.materialize(sizes, sum(sizes.values()))

    def facet(self, ranking, role):
        active = {doc: rank for doc, rank in ranking.items() if self.compiled.eligible[doc]}
        return self.compiled.facet(active, role, self.indexed)


def load(projection, manifest):
    spec = json.loads(manifest.read_bytes())
    if spec["revision"] != previous.live.elife.publisher.REVISION or prior.digest(projection) != spec["projection_sha256"]:
        raise ValueError("Changed publisher projection")
    rows = json.loads(projection.read_bytes())["sources"]
    ownership = {row["doi"]: row for row in rows}
    if len(ownership) != len(rows):
        raise ValueError("Non-unique source identifiers")
    documents = {doi: tuple(row["text"].split("\n", 1)) for doi, row in ownership.items()}
    return previous.live.BASE.pinned.Catalogue(documents, ownership)


def query(index, resources, atoms, tags, arms):
    full, linked, links = resources.full, resources.linked, resources.links
    started = perf_counter()
    ranks = index.rank([topic for topic, _ in atoms], depth=resources.depth)
    anchors, extra = replay.live.simple_order(ranks, depth=resources.depth)
    fallback = [*anchors, extra[0]]
    prefix_ms = (perf_counter() - started) * 1000
    predictions, timings, reach = {}, {}, Counter()

    for arm in arms:
        started = perf_counter()
        lanes = links.filter(ranks) if arm in ("lane_links", "packed_lane") else ranks
        if arm.startswith("packed"):
            facets = tuple(resources.packed.facet(lane, role)
                           for lane, (_, role) in zip(lanes, atoms, strict=True))
        else:
            postings = full if arm == "full" else linked
            facets = tuple(postings.facet(lane, role, 100, True)
                           for lane, (_, role) in zip(lanes, atoms, strict=True))
        plan = replay.typed.source_plan(ranks, facets, fallback, resources.depth)
        planner_ms = (perf_counter() - started) * 1000
        if len(plan) != 3 or len(set(plan)) != 3:
            raise ValueError("Require three distinct sources")
        loaded = index.open_sources(plan)
        proofs = replay.live.BASE.choose_proofs(replay.observed(loaded, atoms, ranks), ranks)
        payload = json.dumps(proofs[:1], separators=(",", ":"))
        timings[arm] = prefix_ms + (perf_counter() - started) * 1000
        match = next(iter(json.loads(payload)), None)

        def tagged(proof):
            return all(topic in tags.get(doc, ()) for (topic, _), doc in zip(atoms, proof[1:], strict=True))

        predictions[arm] = dict(plan=plan, match=match, tagged=bool(match) and tagged(match),
                                rows=sum(map(len, facets)))
        reach[arm] += int(any(tagged(proof) for proof in proofs))
        timings[arm + "_plan"] = planner_ms

    for arm in ARMS[1:]:
        if any(predictions[arm][key] != predictions["full"][key] for key in ("plan", "match", "tagged")):
            raise ValueError("An exact index changed the read plan or response")
    if len(set(reach.values())) != 1:
        raise ValueError("An exact index changed the reachable proof set")
    return predictions, timings


def run_panel(catalogue, requests, tags, embedding, scratch, reverse):
    index = replay.live.LocalIndex(catalogue, previous.mixed.QwenTorchEmbedding(embedding), scratch)
    masks, inventory = certificate.indexes(catalogue)
    encoded = inventory["elife"]
    degrees = Counter(person for owners in catalogue.byline.values() for person in owners)
    encoded["degree_histogram"] = Counter(degrees.values())
    full, linked = masks["full"], masks["linked_compact"]
    links = LaneLinks(full.byline, linked.personal)
    packed = PackedLinks(full)
    result = {}
    for depth in (DEPTHS[::-1] if reverse else DEPTHS):
        resources = SimpleNamespace(full=full, linked=linked, links=links, packed=packed, depth=depth)
        order = ARMS[::-1] if reverse else ARMS
        query(index, resources, requests[0], tags, order)
        ordered = list(enumerate(requests))
        measurements, decisions = {}, {}
        for i, atoms in (reversed(ordered) if reverse else ordered):
            offset = i % len(order)
            decisions[i], measurements[i] = query(index, resources, atoms, tags, order[offset:] + order[:offset])
        decisions = [decisions[i] for i in range(len(requests))]
        result[str(depth)] = dict(
            issued=sum(row["full"]["match"] is not None for row in decisions),
            tagged=sum(row["full"]["tagged"] for row in decisions),
            verified_exact=len(decisions),
            reads_per_arm=3 * len(decisions),
            planning_rows_mean={arm: float(np.mean([row[arm]["rows"] for row in decisions])) for arm in ARMS},
            p95_ms={key: float(np.quantile([timing[key] for timing in measurements.values()], .95))
                    for key in measurements[0]},
            decision_sha256=sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest(),
        )
    return dict(sources=len(catalogue.ids), query_count=len(requests), index=encoded,
                adjacency_payload_bytes=links.payload(), packed_projected_payload_bytes=packed.payload,
                depths=result)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU")
    snapshots = {key: load(getattr(args, key), getattr(args, key + "_manifest")) for key in PANELS}
    one, two = snapshots.values()
    if (set(one.ids) & set(two.ids)
            or set().union(*one.byline.values()) & set().union(*two.byline.values())):
        raise ValueError("Holdout source and verified contributor sets must be disjoint")
    checkpoint = previous.live.verify_embedding(args.embedding)
    result = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        root = Path(scratch)
        for key, catalogue in snapshots.items():
            tags = {doc: {" ".join(topic.casefold().split()) for topic in row["topics"]}
                    for doc, row in catalogue.ownership.items()}
            requests = previous.live.BASE.query_panel(previous.live.BASE.annotation_postings(catalogue, tags))
            if len(requests) < 50:
                raise ValueError("Too few held-out requests")
            result[key] = run_panel(catalogue, requests, tags, args.embedding, root / key, args.reverse)
            linked = certificate.indexes(catalogue)[0]["linked_only"].personal
            linked_catalogue = previous.live.BASE.pinned.Catalogue(
                {doc: catalogue.documents[doc] for doc in linked}, {doc: catalogue.ownership[doc] for doc in linked},
            )
            result[key + "_connected"] = run_panel(
                linked_catalogue, requests, tags, args.embedding, root / (key + "_connected"), args.reverse,
            )
    prior.write_json(args.output, dict(
        panels=result, embedding=checkpoint, script_sha256=prior.digest(Path(__file__)), reverse=args.reverse,
        projection_sha256={key: prior.digest(getattr(args, key)) for key in PANELS},
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("extended", "extended_manifest", "confirmation", "confirmation_manifest", "embedding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
