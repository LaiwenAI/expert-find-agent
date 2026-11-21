"""Preserve cross-source contributor plans with a compact exact role index."""

import argparse
from collections import Counter
from copy import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from experiments.credit import open_topic_index_placement as placement

prior, replay, previous = placement.prior, placement.replay, placement.previous
ARMS = "full", "linked_only", "linked_compact"


class LinkedPostings(replay.typed.RolePostings):
    """Use original ranks while skipping roles on sources that cannot join."""

    def facet(self, ranking, role, coverage, personal):
        active = {doc: rank for doc, rank in ranking.items() if doc in self.personal}
        return super().facet(active, role, coverage, personal)


def indexes(catalogue):
    full = replay.typed.RolePostings(catalogue)
    if any(set(full.personal[doc]) != owners for doc, owners in full.byline.items()):
        raise ValueError("Exact certificate requires identical indexed identity universes")
    degrees = Counter(owner for owners in full.byline.values() for owner in owners)
    linked = {doc for doc, owners in full.byline.items() if any(degrees[e] > 1 for e in owners)}

    masked = copy(full)
    masked.personal = {doc: full.personal[doc] for doc in linked}
    masked.article = {doc: full.article[doc] for doc in linked}
    masked.order = {doc: 0 if doc in linked else 1 << 32 for doc in full.byline}

    compact = LinkedPostings.__new__(LinkedPostings)
    compact.personal, compact.article = masked.personal, masked.article
    compact.order = dict.fromkeys(linked, 0)
    compact.byline = full.byline

    inventory = {}
    for publisher in ("plos", "elife"):
        selected = {doc for doc in catalogue.ids if doc.startswith("10.1371/") == (publisher == "plos")}
        inventory[publisher] = {}
        for arm, index in zip(ARMS[:2], (full, masked), strict=True):
            rows = {doc: {e: sorted(roles) for e, roles in index.personal[doc].items()}
                    for doc in sorted(selected & index.personal.keys())}
            inventory[publisher][arm] = dict(
                documents=len(rows),
                role_bytes=len(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()),
                role_postings=sum(len(roles) for people in rows.values() for roles in people.values()),
            )
        inventory[publisher]["degree_histogram"] = Counter(
            degree for owner, degree in degrees.items() if owner.startswith(f"{publisher}/")
        )
    return dict(zip(ARMS, (full, masked, compact), strict=True)), inventory


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU")
    catalog, panels, model_version, _ = previous.pinned_inputs(args)
    started = perf_counter()
    masks, inventory = indexes(catalog)
    build_ms = (perf_counter() - started) * 1000
    results = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = replay.live.LocalIndex(catalog, previous.mixed.QwenTorchEmbedding(args.embedding),
                                       Path(scratch) / "reads")
        for name, (requests, tags) in panels.items():
            rows, times, reachable = replay.paired.replay(
                index, masks, requests, tags, args.reverse, arms=ARMS, runner=placement.query,
            )
            if any(row["arms"][arm] != row["arms"]["full"] for row in rows for arm in ARMS[1:]):
                raise ValueError("The certified index changed a plan, expert, citation or label")
            if len(set(reachable.values())) != 1:
                raise ValueError("The certified index changed which proofs were reachable")
            results[name] = dict(
                requests=len(rows), reads_per_arm=3 * len(rows),
                known=sum(row["arms"]["full"]["known"] for row in rows),
                issued=sum(row["arms"]["full"]["match"] is not None for row in rows),
                reachable=reachable["full"], exact_plan_and_output_matches=len(rows),
                p95_ms={arm: float(np.quantile(result, .95)) for arm, result in times.items()},
                decision_signature=replay.paired.signature(rows),
            )
    prior.write_json(args.output, dict(
        panels=results, inventory=inventory, build_and_inventory_ms=build_ms,
        script_sha256=prior.digest(Path(__file__)), planner_sha256=prior.digest(Path(placement.__file__)),
        embedding=model_version, reverse=args.reverse, job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
