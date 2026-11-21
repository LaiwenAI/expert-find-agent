"""Compare source ranking signals in fixed-budget, role-aware nomination."""

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

from experiments.credit import open_topic_role_matched as previous
from experiments.followir.core17_reranking import digest, write_json

live, typed, audit = previous.live, previous.typed, previous.audit
DEPTH = previous.DEPTH
TOPIC, SEQUENTIAL, JOINT, LEXICAL, AGREEMENT = "topic", "owner_gap", "joint", "lexical", "agreement"
ARMS = (TOPIC, SEQUENTIAL, JOINT, LEXICAL, AGREEMENT)
COMPARISONS = ((TOPIC, JOINT), (SEQUENTIAL, JOINT), (JOINT, LEXICAL), (JOINT, AGREEMENT),
               (LEXICAL, AGREEMENT))


def planning_ranks(ranks, views, method, size):
    result = []
    for lane, (lexical, dense) in zip(ranks, views, strict=True):
        def priority(doc):
            sparse = lexical.get(doc, size + 1)
            semantic = dense[doc]
            if method == LEXICAL:
                return (sparse, lane[doc])
            return (max(sparse, semantic), sparse + semantic, lane[doc])
        order = sorted(lane, key=priority)
        result.append({doc: at for at, doc in enumerate(order, 1)})
    return tuple(result)


def choose(index, atoms, ranks, plan_ranks, facets, arm):
    if arm == SEQUENTIAL:
        return previous.role_sequential(index, atoms, ranks, facets)
    initial, extras = live.simple_order(ranks, depth=DEPTH)
    default = [*initial, extras[0]]
    opened = default if arm == TOPIC else typed.source_plan(plan_ranks, facets, default, DEPTH)
    return opened, index.open_sources(opened)


def arm_results(index, postings, atoms, subjects, ordered):
    start = perf_counter()
    views = []
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids), views=views)
    ranks = tuple(dict(list(lane.items())[:DEPTH]) for lane in full)
    common_ms = (perf_counter() - start) * 1000
    results, timings, reachable = {}, {}, Counter()
    for arm in ordered:
        start = perf_counter()
        adjusted = planning_ranks(ranks, views, arm, len(index.ids)) if arm in (LEXICAL, AGREEMENT) else ranks
        facets = (() if arm == TOPIC else tuple(postings.facet(lane, role, 100, True)
                                              for lane, (_, role) in zip(ranks, atoms, strict=True)))
        opened, loaded = choose(index, atoms, ranks, adjusted, facets, arm)
        if len(loaded) != 3 or set(opened) != loaded.keys():
            raise ValueError("Expected exactly three distinct reads")
        proofs = live.BASE.choose_proofs(live.BASE.read_articles(
            SimpleNamespace(ownership=loaded), opened, atoms, ranks), ranks)
        payload = json.dumps(proofs[:1], separators=(",", ":"))
        timings[arm] = common_ms + (perf_counter() - start) * 1000
        match = next(iter(json.loads(payload)), None)

        def tagged(proof):
            return all(topic in subjects.get(doi, ())
                       for (topic, _), doi in zip(atoms, proof[1:], strict=True))

        values = [ranks[i][doc] for i, doc in enumerate(match[1:])] if match else []
        results[arm] = dict(match=match, priority=(max(values), sum(values)) if values else None,
                            known=bool(match) and tagged(match), opened=opened)
        reachable[arm] = int(any(tagged(proof) for proof in proofs))
    return dict(atoms=atoms, arms=results, topic_pair=json.dumps([topic for topic, _ in atoms])), timings, reachable


def paired_view(rows, first, second):
    return [row | {"arms": {SEQUENTIAL: row["arms"][first], JOINT: row["arms"][second]}} for row in rows]


def signature(rows):
    return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def replay(index, postings, queries, subjects, reverse, *, arms, runner):
    traces, timing, reachable = [], {arm: [] for arm in arms}, Counter()
    order = arms[::-1] if reverse else arms
    runner(index, postings, queries[0], subjects, order)
    indexed = list(enumerate(queries))
    for i, atoms in (reversed(indexed) if reverse else indexed):
        shift = i % len(order)
        row, times, reach = runner(index, postings, atoms, subjects, order[shift:] + order[:shift])
        traces.append((i, row))
        for arm, duration in times.items():
            timing[arm].append(duration)
        reachable.update(reach)
    return [row for _, row in sorted(traces)], timing, reachable


def compare(rows, comparisons):
    report = {}
    for first, second in comparisons:
        pair = paired_view(rows, first, second)
        cases = {}
        for mode, selection in audit.select(pair).items():
            item = audit.summarize(pair, selection)
            item["arms"] = {first: item["arms"][SEQUENTIAL], second: item["arms"][JOINT]}
            item["issued_only"] = {first: item["issued_only"][SEQUENTIAL], second: item["issued_only"][JOINT]}
            item["possible_known_delta"] = previous.missing_bounds(pair, selection)
            cases[mode] = item
        report[f"{first}_to_{second}"] = cases
    return report


def experiment(index, postings, queries, subjects, reverse, archived):
    rows, timing, reachable = replay(index, postings, queries, subjects, reverse, arms=ARMS, runner=arm_results)
    baseline_signature = signature(paired_view(rows, SEQUENTIAL, JOINT))
    if baseline_signature != archived:
        raise ValueError("Same-role sequential and joint controls must reproduce archived decisions")
    return dict(requests=len(rows), reads_per_arm=3 * len(rows), reachable_after_reads=reachable,
                p95_ms={arm: float(np.quantile(values, .95)) for arm, values in timing.items()},
                comparisons=compare(rows, COMPARISONS), archived_control_signature=baseline_signature, decision_signature=signature(rows))


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and new output")
    catalogue, cases, checkpoint, _ = previous.pinned_inputs(args)
    runs = json.loads(args.archive.read_bytes())["runs"]
    archived = {key: runs[0]["panels"][key]["decision_signature"] for key in cases}
    if any(run["embedding"] != checkpoint or run["reference_sha256"] != digest(args.reference)
           or {key: run["panels"][key]["decision_signature"] for key in cases} != archived for run in runs):
        raise ValueError("Archive runs disagree with required inputs")
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = live.LocalIndex(catalogue, previous.mixed.QwenTorchEmbedding(args.embedding), Path(scratch) / "reads")
        postings = typed.RolePostings(catalogue)
        panels = {key: experiment(index, postings, questions, tags, args.reverse, archived[key])
                  for key, (questions, tags) in cases.items()}
    write_json(args.output, dict(
        panels=panels, archive_sha256=digest(args.archive), script_sha256=digest(Path(__file__)),
        ranker_sha256=digest(Path(live.__file__)), embedding=checkpoint, reverse=args.reverse,
        gpu=torch.cuda.get_device_name(), slurm_job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                  "archive", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
