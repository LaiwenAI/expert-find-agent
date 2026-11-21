"""Test whether confirmed anchor roles improve a fixed three-read evidence budget."""

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from experiments.credit import open_topic_three_source_plan as prior

JOINT, SPLIT, THIRD, SECOND_THIRD = "joint", "split_joint", "repair_third", "repair_second_third"
UPGRADE = "repair_prefer_indexed"
ARMS = (prior.SEQUENTIAL, JOINT, SPLIT, THIRD, SECOND_THIRD, UPGRADE)
COMPARISONS = ((JOINT, SPLIT), (SPLIT, THIRD), (THIRD, SECOND_THIRD),
               (JOINT, SECOND_THIRD), (JOINT, UPGRADE), (SECOND_THIRD, UPGRADE))
live, typed, paired, previous = prior.live, prior.typed, prior.paired, prior.previous


def observed(loaded, atoms, ranks):
    return live.BASE.read_articles(SimpleNamespace(ownership=loaded), loaded, atoms, ranks)


def next_source(people, ranks, facets, loaded, fallback, *, prefer_indexed):
    """Keep an indexed compatible fallback, else repair from confirmed anchor roles."""
    if not people or live.BASE.choose_proofs(people, ranks):
        return fallback

    fallback_levels = {
        level
        for person, lanes in people.items() for side, sources in enumerate(lanes)
        if sources and fallback in facets[1 - side]
        for level, owners in enumerate(facets[1 - side][fallback]) if person in owners
    }
    if 0 in fallback_levels or (fallback_levels and not prefer_indexed):
        return fallback

    # Keep confirmed matches ahead of opaque matches. Use the same two ranks
    # as the final proof selector so a repair does not favor popular owners.
    best = None
    for person, lanes in people.items():
        for side, sources in enumerate(lanes):
            if not sources:
                continue
            anchor_rank = min(ranks[side][doi] for doi in sources)
            other = 1 - side
            for candidate, (known, unknown) in facets[other].items():
                if candidate in loaded or (person not in known and person not in unknown):
                    continue
                candidate_rank = ranks[other][candidate]
                key = (person not in known, max(anchor_rank, candidate_rank),
                       anchor_rank + candidate_rank, candidate)
                if best is None or key < best:
                    best = key
    if best is None or (fallback_levels and best[0]):
        return fallback
    return best[-1]


def read_plan(index, plan, atoms, ranks, facets, arm):
    if arm == JOINT:
        return plan, index.open_sources(plan)
    if arm == SPLIT:
        return plan, index.open_sources(plan[:2]) | index.open_sources(plan[2:])
    prefer_indexed = arm == UPGRADE
    if arm in (SECOND_THIRD, UPGRADE):
        loaded = index.open_sources(plan[:1])
        second = next_source(observed(loaded, atoms, ranks), ranks, facets, loaded, plan[1],
                             prefer_indexed=prefer_indexed)
        loaded |= index.open_sources([second])
    else:
        second = plan[1]
        loaded = index.open_sources(plan[:2])
    fallback = plan[2] if plan[2] not in loaded else plan[1]
    third = next_source(observed(loaded, atoms, ranks), ranks, facets, loaded, fallback,
                        prefer_indexed=prefer_indexed)
    return [plan[0], second, third], loaded | index.open_sources([third])


def query(index, postings, atoms, tags, arms, *, coverage):
    start = perf_counter()
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids))
    ranks = tuple(dict(list(lane.items())[:prior.DEPTH]) for lane in full)
    prefix = perf_counter() - start
    results, timings, reach = {}, {}, Counter()

    def tagged(proof):
        return all(topic in tags.get(doi, ()) for (topic, _), doi in zip(atoms, proof[1:], strict=True))

    for arm in arms:
        start = perf_counter()
        facets = tuple(postings.facet(lane, role, coverage, True)
                       for lane, (_, role) in zip(ranks, atoms, strict=True))
        if arm == prior.SEQUENTIAL:
            opened, loaded = previous.role_sequential(index, atoms, ranks, facets)
        else:
            initial, fresh = live.simple_order(ranks, depth=prior.DEPTH)
            plan = typed.source_plan(ranks, facets, [*initial, fresh[0]], prior.DEPTH)
            opened, loaded = read_plan(index, plan, atoms, ranks, facets, arm)
        if len(opened) != 3 or set(opened) != loaded.keys():
            raise ValueError("Require three distinct declaration reads")
        proofs = live.BASE.choose_proofs(observed(loaded, atoms, ranks), ranks)
        payload = json.dumps(proofs[:1], separators=(",", ":"))
        timings[arm] = (prefix + perf_counter() - start) * 1000
        match = next(iter(json.loads(payload)), None)
        positions = [ranks[i][doc] for i, doc in enumerate(match[1:])] if match else []
        results[arm] = dict(match=match, priority=(max(positions), sum(positions)) if positions else None,
                            known=bool(match) and tagged(match), opened=opened)
        reach[arm] = int(any(tagged(proof) for proof in proofs))
    return dict(atoms=atoms, arms=results, topic_pair=json.dumps([topic for topic, _ in atoms])), timings, reach


def evaluate(index, postings, requests, tags, coverage, reverse, archived):
    rows, times, reached = paired.replay(index, postings, requests, tags, reverse, arms=ARMS,
                                         runner=partial(query, coverage=coverage))
    control = paired.signature(paired.paired_view(rows, prior.SEQUENTIAL, JOINT))
    if coverage == 100 and control != archived:
        raise ValueError("Archived sequential and joint policies must reproduce")
    if any(row["arms"][JOINT] != row["arms"][SPLIT] for row in rows):
        raise ValueError("Splitting a static plan must preserve all decisions")
    plans_changed = {
        arm: sum(row["arms"][arm]["opened"] != row["arms"][JOINT]["opened"] for row in rows)
        for arm in (THIRD, SECOND_THIRD, UPGRADE)
    }
    return dict(
        n=len(rows), reads={arm: 3 * len(rows) for arm in ARMS}, reachable_after_reads=reached,
        changed_plans=plans_changed,
        changed_second_reads={
            arm: sum(row["arms"][arm]["opened"][1] != row["arms"][JOINT]["opened"][1] for row in rows)
            for arm in (SECOND_THIRD, UPGRADE)
        },
        comparisons=paired.compare(rows, COMPARISONS),
        p95_ms={arm: float(np.quantile(values, .95)) for arm, values in times.items()},
        decision_signature=paired.signature(rows), control_signature=control,
    )


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU")
    catalogue, panels, checkpoint, _ = previous.pinned_inputs(args)
    archive = json.loads(args.archive.read_bytes())["runs"]
    signatures = {name: archive[0]["panels"][name]["decision_signature"] for name in panels}
    if any(run["embedding"] != checkpoint or run["reference_sha256"] != prior.digest(args.reference)
           or {name: run["panels"][name]["decision_signature"] for name in panels} != signatures for run in archive):
        raise ValueError("Archive runs disagree with the selected inputs")
    result = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = live.LocalIndex(catalogue, previous.mixed.QwenTorchEmbedding(args.embedding), Path(scratch) / "reads")
        postings = typed.RolePostings(catalogue)
        for coverage in ((25, 50, 100) if args.reverse else (100, 50, 25)):
            for name, (requests, tags) in panels.items():
                result[f"{name}_{coverage}"] = evaluate(
                    index, postings, requests, tags, coverage, args.reverse, signatures[name],
                )
    prior.write_json(args.output, dict(
        panels=result, archive_sha256=prior.digest(args.archive), embedding=checkpoint,
        script_sha256=prior.digest(Path(__file__)), reverse=args.reverse,
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                  "archive", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
