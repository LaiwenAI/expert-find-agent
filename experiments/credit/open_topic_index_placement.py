"""Allocate scarce contributor-role extraction to same-person source pairs."""

import argparse
from collections import Counter, defaultdict
from copy import copy
from itertools import combinations
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from experiments.credit import open_topic_observed_repair as replay

prior, previous = replay.prior, replay.previous
ARMS = ("hash", "authors", "owner_degree", "pair_cover", "diverse_pair")
COMPARISONS = tuple(("hash", arm) for arm in ARMS[1:]) + (
    ("owner_degree", "pair_cover"), ("pair_cover", "diverse_pair"),
)


def pair_placement(byline, titles, anchor_order, costs, budget, *, diversify):
    """Greedily buy source pairs by new two-source owners per extracted record."""
    sources = defaultdict(list)
    for doi in sorted(byline):
        for owner in byline[doi]:
            sources[owner].append(doi)
    pairs = sorted({pair for docs in sources.values() for pair in combinations(docs, 2)})
    words = {doi: set(re.findall(r"[a-z0-9]{3,}", title.lower()))
             for doi, title in titles.items()}

    def lexical_distance(first, second):
        union = words[first] | words[second]
        return 1 - len(words[first] & words[second]) / len(union) if union else 0.

    actions = [(pair, lexical_distance(*pair) if diversify else 0.) for pair in pairs]
    selected, seen = set(), Counter()
    remaining = budget
    while remaining:
        best = None
        for pair, distance in actions:
            extra = tuple(doc for doc in pair if doc not in selected)
            cost = sum(costs[doc] for doc in extra)
            if not extra or cost > remaining:
                continue
            added = Counter(owner for doc in extra for owner in byline[doc])
            closed = sum(seen[owner] < 2 <= seen[owner] + count for owner, count in added.items())
            if closed == 0:
                continue
            introduced = sum(not seen[owner] for owner in added)
            key = (-closed / cost, -distance, -introduced / cost, extra)
            if best is None or key < best:
                best = key
        if best is None:
            break
        for doc in best[-1]:
            selected.add(doc)
            seen.update(byline[doc])
            remaining -= costs[doc]
    return fill(anchor_order, costs, remaining, selected)


def fill(ordered, costs, budget, selected=None):
    selected = set() if selected is None else selected
    for doc in ordered:
        if doc not in selected and costs[doc] <= budget:
            selected.add(doc)
            budget -= costs[doc]
    return selected


def choose(catalogue, docs, costs, budget, original, arm):
    if arm == "hash":
        return fill(sorted(docs, key=lambda doi: (original.order[doi], doi)), costs, budget)
    byline = {doi: catalogue.byline[doi] for doi in docs}
    if arm == "authors":
        ordered = sorted(docs, key=lambda doi: (-len(byline[doi]) / costs[doi], doi))
        return fill(ordered, costs, budget)
    degrees = Counter(owner for owners in byline.values() for owner in owners)
    ordered = sorted(docs, key=lambda doi: (
        -sum(np.log2(degrees[o]) for o in sorted(byline[doi])) / costs[doi], -len(byline[doi]), doi,
    ))
    if arm == "owner_degree":
        return fill(ordered, costs, budget)
    titles = {doi: catalogue.documents[doi][0] for doi in docs}
    return pair_placement(byline, titles, ordered, costs, budget, diversify=arm == "diverse_pair")


def build_placements(catalogue, original, partition, coverage, *, byte_matched):
    cutoff = coverage * (1 << 32) // 100
    payload = {doi: {person: sorted(roles) for person, roles in original.personal[doi].items()}
               for doi in catalogue.ids}
    # The per-record comma is charged for every entry. A nonempty JSON
    # object adds one further byte overall, independent of record count.
    record_bytes = {doi: len(json.dumps({doi: row}, separators=(",", ":")).encode()) - 1
                    for doi, row in payload.items()}
    costs = record_bytes if byte_matched else dict.fromkeys(catalogue.ids, 1)
    budgets = {name: sum(costs[doi] for doi in docs if original.order[doi] < cutoff)
               for name, docs in partition.items()}
    masks, inventories = {}, {}
    arms = (*ARMS, "full") if coverage == 25 else ARMS
    for arm in arms:
        started = perf_counter()
        selection = (set(catalogue.ids) if arm == "full" else set().union(*(
            choose(catalogue, docs, costs, budgets[name], original, arm)
            for name, docs in partition.items()
        )))
        placements_ms = (perf_counter() - started) * 1000
        postings = copy(original)
        postings.order = {doi: 0 if doi in selection else 1 << 32 for doi in catalogue.ids}
        masks[arm] = postings
        inventories[arm] = {}
        for name, docs in partition.items():
            chosen = sorted(selection & docs)
            cost = sum(costs[doi] for doi in chosen)
            capacity = sum(costs[doi] for doi in docs) if arm == "full" else budgets[name]
            if cost > capacity or (not byte_matched and cost != capacity):
                raise ValueError("Extraction or index byte budget violated")
            owners = Counter(owner for doi in chosen for owner in catalogue.byline[doi])
            inventories[arm][name] = dict(
                capacity=capacity, spent=cost, sources=len(chosen),
                owners_with_two_sources=sum(count >= 2 for count in owners.values()),
                role_postings=sum(len(roles) for doi in chosen for roles in payload[doi].values()),
                encoded_role_bytes=1 + sum(record_bytes[doi] for doi in chosen) if chosen else 2,
                source_set_sha256=sha256(json.dumps(chosen).encode()).hexdigest(),
            )
        inventories[arm]["offline_selection_ms"] = placements_ms
    return masks, inventories


def query(index, masks, atoms, tags, arms):
    started = perf_counter()
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids))
    ranks = tuple(dict(list(lane.items())[:prior.DEPTH]) for lane in full)
    prefix = perf_counter() - started
    results, timings, reachable = {}, {}, Counter()

    def tagged(proof):
        return all(topic in tags.get(doi, ()) for (topic, _), doi in zip(atoms, proof[1:], strict=True))

    for arm in arms:
        started = perf_counter()
        facets = tuple(masks[arm].facet(lane, role, 100, True)
                       for lane, (_, role) in zip(ranks, atoms, strict=True))
        anchors, extras = replay.live.simple_order(ranks, depth=prior.DEPTH)
        plan = replay.typed.source_plan(ranks, facets, [*anchors, extras[0]], prior.DEPTH)
        loaded = index.open_sources(plan)
        if len(plan) != 3 or loaded.keys() != set(plan):
            raise ValueError("Every route must read exactly three distinct sources")
        proofs = replay.live.BASE.choose_proofs(replay.observed(loaded, atoms, ranks), ranks)
        payload = json.dumps(proofs[:1], separators=(",", ":"))
        timings[arm] = (prefix + perf_counter() - started) * 1000
        match = next(iter(json.loads(payload)), None)
        positions = [ranks[i][doc] for i, doc in enumerate(match[1:])] if match else []
        results[arm] = dict(match=match, priority=(max(positions), sum(positions)) if positions else None,
                            known=bool(match) and tagged(match), opened=plan)
        reachable[arm] += any(tagged(proof) for proof in proofs)
    return dict(atoms=atoms, arms=results, topic_pair=json.dumps([topic for topic, _ in atoms])), timings, reachable


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU")
    catalogue, panels, checkpoint, _ = previous.pinned_inputs(args)
    archived = json.loads(args.archive.read_bytes())["results"]["panels"]
    partition = {"plos": {doi for doi in catalogue.ids if doi.startswith("10.1371/")}}
    partition["elife"] = set(catalogue.ids) - partition["plos"]
    if [len(docs) for docs in partition.values()] != [130, 322]:
        raise ValueError("Unexpected publisher populations")
    results, inventory = {}, {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = replay.live.LocalIndex(catalogue, previous.mixed.QwenTorchEmbedding(args.embedding), Path(scratch) / "reads")
        original = replay.typed.RolePostings(catalogue)
        settings = [(byte_matched, coverage) for byte_matched in (False, True)
                    for coverage in (2, 5, 10, 25, 50)]
        for byte_matched, coverage in (reversed(settings) if args.reverse else settings):
            mode = "bytes" if byte_matched else "records"
            masks, inventory[f"{mode}_{coverage}"] = build_placements(
                catalogue, original, partition, coverage, byte_matched=byte_matched,
            )
            for name, (requests, tags) in panels.items():
                key = f"{name}_{coverage}"
                arms = tuple(masks)
                comparisons = COMPARISONS
                if "full" in masks:
                    comparisons += tuple(("full", arm) for arm in ("hash", "owner_degree", "pair_cover"))
                rows, times, reachable = replay.paired.replay(
                    index, masks, requests, tags, args.reverse, arms=arms, runner=query,
                )
                if key in archived:
                    expected = archived[key]["comparisons"]["joint_to_split_joint"]["natural"]["arms"]["joint"]
                    baseline = {
                        "issued": sum(row["arms"]["hash"]["match"] is not None for row in rows),
                        "known": sum(row["arms"]["hash"]["known"] for row in rows),
                    }
                    if (any(baseline[field] != expected[field] for field in baseline)
                            or reachable["hash"] != archived[key]["reachable_after_reads"]["joint"]):
                        raise ValueError("Archived hash-masked control did not reproduce")
                results[f"{mode}_{key}"] = dict(
                    requests=len(rows), reads_per_arm=3 * len(rows), reachable_after_reads=reachable,
                    comparisons=replay.paired.compare(rows, comparisons),
                    p95_ms={arm: float(np.quantile(values, .95)) for arm, values in times.items()},
                    decision_signature=replay.paired.signature(rows),
                )
    prior.write_json(args.output, dict(
        panels=results, inventory=inventory, embedding=checkpoint, reverse=args.reverse,
        archive_sha256=prior.digest(args.archive), script_sha256=prior.digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                  "archive", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
