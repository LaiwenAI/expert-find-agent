"""Validate and apply source relevance scores before three declaration reads."""

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

from expert_search.statistics import clustered_paired_interval
from experiments.credit import open_topic_joint_agreement as paired
from experiments.credit import open_topic_role_matched as previous
from experiments.credit import open_topic_three_source_plan as triple
from experiments.csfcube.frozen_confirmation import FACETS, load
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text

live, typed = previous.live, previous.typed
DEPTH = previous.DEPTH
JOINT, RANK, SIX, FULL = "joint", "all_edges", "reader_six", "reader_full"
ARMS = (JOINT, RANK, SIX, FULL)
COMPARISONS = ((JOINT, RANK), (JOINT, SIX), (JOINT, FULL), (RANK, FULL), (SIX, FULL))
CAP = 6
CHUNK = 12


def reader_scores(reader, pairs, task):
    output = []
    clipped = calls = 0
    for start in range(0, len(pairs), CHUNK):
        scores, truncated = reader.score_pairs(pairs[start:start + CHUNK], task=task)
        output.extend(scores)
        clipped += truncated
        calls += 1
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite source scores")
    return output, clipped, calls


def public_probe(data, encoder, reader, reverse):
    papers, questions, hashes = load(data)
    documents = {row["corpusid"]: paper_text(row) for row in papers}
    ids = {key: i for i, key in enumerate(documents)}
    matrix = encoder.encode(list(documents.values()))
    requests = [row["query"] for row in questions]
    vectors = encoder.encode(requests, query=True)
    records = []
    ordered = range(len(questions) - 1, -1, -1) if reverse else range(len(questions))
    for i in ordered:
        row = questions[i]
        query_text = documents[row["query_id"]]
        grades = row["grades"]
        choices = sorted(grades)
        similarities = matrix[[ids[key] for key in choices]] @ vectors[i]
        indices = sorted(range(len(choices)), key=lambda j: (-similarities[j], choices[j]))[:6]
        pool = [choices[j] for j in indices]
        task = f"Decide whether this article has a similar {FACETS[row['facet']]} to the query article."
        start = perf_counter()
        scores, truncated, _ = reader_scores(reader, [(query_text, documents[key]) for key in pool], task)
        elapsed = (perf_counter() - start) * 1000
        ranking = sorted(range(len(pool)), key=lambda j: (-scores[j], j))
        comparisons = [(a, b) for a, b in product(range(len(pool)), repeat=2)
                       if grades[pool[a]] >= 2 > grades[pool[b]]]
        concordant = [sum((.5 if scores[a] == scores[b] else float(scores[a] > scores[b]))
                          if mode == "reader" else float(a < b) for a, b in comparisons)
                      for mode in ("dense", "reader")]
        records.append((i, row["query_id"], grades[pool[0]], grades[pool[ranking[0]]],
                        len(comparisons), concordant, truncated, elapsed))
    records.sort()
    return dict(
        requests=len(records), input_sha256=hashes,
        known_strong_top1={name: sum(row[2+j] >= 2 for row in records) for j, name in enumerate(("dense", "reader"))},
        paired_strong_top1_delta=clustered_paired_interval(
            [int(row[3] >= 2) - int(row[2] >= 2) for row in records], [row[1] for row in records]),
        grade_pairs=sum(row[4] for row in records),
        correctly_ordered_grade_pairs=[sum(row[5][j] for row in records) for j in range(2)],
        clipped_pairs=sum(row[6] for row in records),
        reranker_stage_p95_ms=float(np.quantile([row[7] for row in records], .95)),
    )


def orientations(facets, ranks):
    owners = defaultdict(lambda: (set(), set()))
    for side, lane in enumerate(facets):
        for doi, (known, opaque) in lane.items():
            for person in known | opaque:
                owners[person][side].add(doi)
    return sorted({(a, b) for left, right in owners.values() for a, b in product(left, right) if a != b},
                  key=lambda edge: (max(ranks[0][edge[0]], ranks[1][edge[1]]),
                                    ranks[0][edge[0]] + ranks[1][edge[1]], edge))


def slots(edge):
    return {(0, edge[0]), (1, edge[1])}


def choose(reader, atoms, index, edges, candidates, original, rank_plan, mode):
    # Score only when the plan can change. Preserve the original plan on ties.
    if len(candidates) <= 1 or len(edges) < 2:
        return rank_plan, Counter()
    if mode == SIX:
        allowed = set()
        preferred = [edge for edge in edges if set(edge) <= set(original)]
        for edge in [*preferred, *edges]:
            proposed = allowed | slots(edge)
            if len(proposed) <= CAP:
                allowed = proposed
        edges = [edge for edge in edges if slots(edge) <= allowed]
    else:
        allowed = set().union(*(slots(edge) for edge in edges))
    if len(edges) < 2:
        return rank_plan, Counter()

    ordered = sorted(allowed)
    inputs = [(atoms[side][0], index.texts[doi]) for side, doi in ordered]
    raw, clipped, calls = reader_scores(reader, inputs, live.TOPIC_TASK)
    scores = dict(zip(ordered, raw, strict=True))
    weights = defaultdict(lambda: -np.inf)
    for a, b in edges:
        pair = tuple(sorted((a, b)))
        weights[pair] = max(weights[pair], min(scores[0, a], scores[1, b]))
    anchor = set(original)

    def priority(docs):
        return (-max(weights[pair] for pair in combinations(docs, 2)), set(docs) != anchor, docs)

    return min(candidates, key=priority), Counter(model_calls=calls, pairs=len(inputs), clipped_pairs=clipped)


def query(index, postings, atoms, subjects, order, *, reader):
    start = perf_counter()
    full = index.rank([topic for topic, _ in atoms], depth=len(index.ids))
    ranks = tuple(dict(list(lane.items())[:DEPTH]) for lane in full)
    recall_ms = (perf_counter() - start) * 1000
    results, times, reach = {}, {}, Counter()
    for arm in order:
        start = perf_counter()
        facets = tuple(postings.facet(lane, role, 100, True) for lane, (_, role) in zip(ranks, atoms, strict=True))
        initial, extras = live.simple_order(ranks, depth=DEPTH)
        original = typed.source_plan(ranks, facets, [*initial, extras[0]], DEPTH)
        candidate, work = original, Counter()
        if arm != JOINT:
            rank_plan, candidates = triple.triple_plan(ranks, facets, original, triple.ALL_EDGES)
            candidate = rank_plan
            if arm in (SIX, FULL):
                candidate, work = choose(reader, atoms, index, orientations(facets, ranks), candidates,
                                         original, rank_plan, arm)
        opened = original if set(candidate) == set(original) else list(candidate)
        loaded = index.open_sources(opened)
        if len(loaded) != 3 or set(opened) != loaded.keys():
            raise ValueError("Expected three distinct reads")
        proofs = live.BASE.choose_proofs(live.BASE.read_articles(
            SimpleNamespace(ownership=loaded), opened, atoms, ranks), ranks)
        payload = json.dumps(proofs[:1], separators=(",", ":"))
        times[arm] = recall_ms + (perf_counter() - start) * 1000
        match = next(iter(json.loads(payload)), None)
        values = [ranks[i][doi] for i, doi in enumerate(match[1:])] if match else []

        def tagged(proof):
            return all(topic in subjects.get(doi, ()) for (topic, _), doi in zip(atoms, proof[1:], strict=True))

        results[arm] = dict(match=match, priority=(max(values), sum(values)) if values else None,
                            known=bool(match) and tagged(match), opened=opened, work=work)
        reach[arm] = int(any(tagged(proof) for proof in proofs))
    return dict(atoms=atoms, arms=results, topic_pair=json.dumps([topic for topic, _ in atoms])), times, reach


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on an allocated CUDA device")
    catalogue, panels, checkpoint, _ = previous.pinned_inputs(args)
    reader = Reranker(args.reranker)
    encoder = previous.mixed.QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        index = live.LocalIndex(catalogue, encoder, Path(temp) / "reads")
        postings = typed.RolePostings(catalogue)
        output = {}
        for name, (questions, tags) in panels.items():
            observations, times, reach = paired.replay(index, postings, questions, tags, args.reverse,
                                                       arms=ARMS, runner=partial(query, reader=reader))
            calls = {arm: sum((row["arms"][arm]["work"] for row in observations), Counter()) for arm in ARMS}
            comparisons = paired.compare(observations, COMPARISONS)
            archived = json.loads(args.archive.read_bytes())["panels"][f"{name}_100"]
            if comparisons["joint_to_all_edges"] != archived["comparisons"]["joint_to_all_edges"]:
                raise ValueError("The original rank-only results changed")
            output[name] = dict(
                comparisons=comparisons, signature=paired.signature(observations),
                reachable_after_reads=reach, reader_work=calls,
                p95_ms={arm: float(np.quantile(values, .95)) for arm, values in times.items()})
            print(name, calls, flush=True)
    public = public_probe(args.public, encoder, reader, args.reverse)
    write_json(args.output, dict(
        panels=output, public=public, embedding=checkpoint, reranker=reader.provenance["revision"],
        archive_sha256=digest(args.archive), script_sha256=digest(Path(__file__)),
        reverse=args.reverse, gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                  "archive", "reranker", "public", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
