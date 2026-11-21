"""Compare anchored and joint three-source selection under a fixed read budget."""

import argparse
from collections import Counter, defaultdict
from functools import partial
from itertools import combinations, product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from experiments.credit import open_topic_joint_agreement as paired
from experiments.credit import open_topic_role_matched as previous
from experiments.followir.core17_reranking import digest, write_json

live, typed = previous.live, previous.typed
DEPTH = previous.DEPTH
SEQUENTIAL, JOINT = "owner_gap", "joint"
ANCHORED, FREE, ALL_EDGES = "anchored", "free", "all_edges"
OPTIMIZED = (ANCHORED, FREE, ALL_EDGES)
ARMS = (SEQUENTIAL, JOINT, *OPTIMIZED)
COMPARISONS = ((JOINT, ANCHORED), (JOINT, FREE), (JOINT, ALL_EDGES), (FREE, ALL_EDGES))
SEEDS = 16


def weighted_edges(ranks, facets):
    owners = defaultdict(lambda: ([], []))
    for side, supports in enumerate(facets):
        for doi, (known, opaque) in supports.items():
            weight = 60 / (60 + ranks[side][doi])
            for person in known:
                owners[person][side].append((doi, weight))
            for person in opaque:
                owners[person][side].append((doi, .5 * weight))
    edges = {}
    for first, second in owners.values():
        for (a, weight_a), (b, weight_b) in product(first, second):
            if a != b:
                pair = tuple(sorted((a, b)))
                edges[pair] = max(edges.get(pair, 0.), weight_a * weight_b)
    return edges


def triple_plan(ranks, facets, original, arm):
    edges = weighted_edges(ranks, facets)
    neighbors = defaultdict(set)
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    anchor = tuple(sorted(original[:2]))
    seeds = [anchor]
    if arm == FREE:
        seeds.extend(sorted(edges, key=lambda pair: (-edges[pair], pair))[:SEEDS])
    elif arm == ALL_EDGES:
        seeds.extend(edges)
        fillers = sorted(set().union(*ranks), key=lambda doi: (
            min(lane.get(doi, DEPTH + 1) for lane in ranks), doi,
        ))[:3]

    candidate_sets = {tuple(sorted(original))}
    for a, b in seeds:
        extra = (neighbors[a] | neighbors[b]) - {a, b}
        if arm == ALL_EDGES and not extra:
            extra = {next(c for c in fillers if c not in {a, b})}
        for c in extra:
            candidate_sets.add(tuple(sorted((a, b, c))))

    def priority(documents):
        edge_scores = sorted((edges.get(pair, 0.) for pair in combinations(documents, 2)), reverse=True)
        return (-(edge_scores[0] + .5 * edge_scores[1]), -edge_scores[0],
                sum(min(lane.get(doi, DEPTH + 1) for lane in ranks) for doi in documents), documents)

    return min(candidate_sets, key=priority), candidate_sets


def label_supported_pairs(postings, atoms, ranks, subjects):
    owners = defaultdict(lambda: ([], []))
    for side, ((topic, role), lane) in enumerate(zip(atoms, ranks, strict=True)):
        for doi in lane:
            if topic in subjects.get(doi, ()):
                for person, duties in postings.personal[doi].items():
                    if role in duties:
                        owners[person][side].append(doi)
    return {tuple(sorted((a, b))) for first, second in owners.values() for a, b in product(first, second) if a != b}


def query(index, postings, atoms, subjects, order, *, coverage):
    started = perf_counter()
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids))
    ranks = tuple(dict(list(lane.items())[:DEPTH]) for lane in full)
    prefix_ms = (perf_counter() - started) * 1000
    results, timings, reach = {}, {}, Counter()

    def tagged(proof):
        return all(topic in subjects.get(doi, ()) for (topic, _), doi in zip(atoms, proof[1:], strict=True))

    ceilings = {}
    for arm in order:
        started = perf_counter()
        facets = tuple(postings.facet(lane, role, coverage, True)
                       for lane, (_, role) in zip(ranks, atoms, strict=True))
        evaluated = None
        if arm == SEQUENTIAL:
            opened, loaded = previous.role_sequential(index, atoms, ranks, facets)
        else:
            initial, extras = live.simple_order(ranks, depth=DEPTH)
            original = typed.source_plan(ranks, facets, [*initial, extras[0]], DEPTH)
            opened = original
            if arm in OPTIMIZED:
                candidate, evaluated = triple_plan(ranks, facets, original, arm)
                if set(candidate) != set(original):
                    opened = list(candidate)
            loaded = index.open_sources(opened)
        if len(loaded) != 3 or set(opened) != loaded.keys():
            raise ValueError("Every policy must read exactly three distinct sources")
        proofs = live.BASE.choose_proofs(live.BASE.read_articles(
            SimpleNamespace(ownership=loaded), opened, atoms, ranks), ranks)
        serialized = json.dumps(proofs[:1], separators=(",", ":"))
        timings[arm] = prefix_ms + (perf_counter() - started) * 1000
        match = next(iter(json.loads(serialized)), None)
        values = [ranks[i][doi] for i, doi in enumerate(match[1:])] if match else []
        results[arm] = dict(match=match, priority=(max(values), sum(values)) if values else None,
                            known=bool(match) and tagged(match), opened=opened)
        if evaluated is not None:
            results[arm]["candidate_sets"] = len(evaluated)
            ceilings[arm] = evaluated
        reach[arm] = int(any(tagged(proof) for proof in proofs))
    expected = label_supported_pairs(postings, atoms, ranks, subjects)
    opportunities = {arm: int(any(pair in expected for docs in sets for pair in combinations(docs, 2)))
                     for arm, sets in ceilings.items()}
    return dict(atoms=atoms, arms=results, topic_pair=json.dumps([topic for topic, _ in atoms]),
                target_pool=bool(expected), frontier=opportunities), timings, reach


def evaluate(index, postings, queries, tags, coverage, archived, reverse):
    rows, timings, reach = paired.replay(index, postings, queries, tags, reverse, arms=ARMS,
                                         runner=partial(query, coverage=coverage))
    baseline = paired.signature(paired.paired_view(
        [{key: row[key] for key in ("atoms", "arms", "topic_pair")} for row in rows], SEQUENTIAL, JOINT))
    ceiling = {arm: sum(row["frontier"][arm] for row in rows) for arm in OPTIMIZED}
    if any(ceiling[arm] < reach[arm] for arm in ceiling):
        raise ValueError("Chosen plans exceed the available evidence frontier")
    if coverage == 100 and baseline != archived:
        raise ValueError("The historical full-coverage control did not reproduce")
    return dict(
        requests=len(rows), reads_per_arm=3 * len(rows), role_index_coverage=coverage,
        index_counts=postings.inventory().get(str(coverage)),
        reachable_after_reads=reach,
        tag_supported_in_pool=sum(row["target_pool"] for row in rows),
        tag_supported_in_frontier=ceiling,
        changed_plans={arm: sum(set(row["arms"][arm]["opened"]) != set(row["arms"][JOINT]["opened"])
                               for row in rows) for arm in OPTIMIZED},
        candidate_sets_p50_p95={arm: np.quantile([row["arms"][arm]["candidate_sets"] for row in rows],
                                               [.5, .95]).tolist() for arm in OPTIMIZED},
        p95_ms={arm: float(np.quantile(values, .95)) for arm, values in timings.items()},
        comparisons=paired.compare(rows, COMPARISONS),
        decision_signature=paired.signature(rows), baseline_signature=baseline,
    )


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require new output on a Slurm CUDA node")
    catalogue, panels, checkpoint, _ = previous.pinned_inputs(args)
    archive = json.loads(args.archive.read_bytes())["runs"]
    signatures = {name: archive[0]["panels"][name]["decision_signature"] for name in panels}
    if any(run["embedding"] != checkpoint or run["reference_sha256"] != digest(args.reference)
           or {name: run["panels"][name]["decision_signature"] for name in panels} != signatures for run in archive):
        raise ValueError("Archive run mismatch")
    results = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = live.LocalIndex(catalogue, previous.mixed.QwenTorchEmbedding(args.embedding), Path(scratch) / "reads")
        postings = typed.RolePostings(catalogue)
        for coverage in ((50, 100) if args.reverse else (100, 50)):
            for name, (requests, tags) in panels.items():
                results[f"{name}_{coverage}"] = evaluate(index, postings, requests, tags, coverage,
                                                        signatures[name], args.reverse)
    write_json(args.output, dict(
        panels=results, archive_sha256=digest(args.archive), script_sha256=digest(Path(__file__)),
        embedding=checkpoint, reverse=args.reverse, gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                "archive", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
