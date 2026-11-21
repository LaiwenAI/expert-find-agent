"""Test whether inexpensive source evidence scores transfer across publishers."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from experiments.credit import linked_role_lane_index as linked
from experiments.credit import open_topic_three_source_plan as plans


ARMS = ("joint", "rank_set", "learned_set")
DEPTH = 128
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def terms(text):
    return set(WORD.findall(text.casefold()))


def features(title, body, topic, rank, sparse, dense):
    keys = terms(topic)
    scale = max(len(keys), 1)
    return (
        60 / (60 + rank),
        60 / (60 + dense),
        60 / (60 + sparse) if sparse else 0.,
        len(keys & title) / scale,
        len(keys & body) / scale,
        float(bool(sparse)),
    )


def candidates(index, topics, postings, atoms):
    start = perf_counter()
    views = []
    ranks = index.rank([topic for topic, _ in atoms], depth=DEPTH, views=views)
    facets = tuple(postings.facet(lane, role, 100, True)
                   for lane, (_, role) in zip(ranks, atoms, strict=True))
    anchors, extra = plans.live.simple_order(ranks, depth=DEPTH)
    baseline = plans.typed.source_plan(ranks, facets, [*anchors, extra[0]], DEPTH)
    common = perf_counter() - start
    extra_started = perf_counter()
    scores = plans.weighted_edges(ranks, facets)
    pool = {doc for pair in scores for doc in pair}
    side_features = []
    oriented = {}
    for side, ((topic, _), lane, (sparse, dense)) in enumerate(zip(atoms, ranks, views, strict=True)):
        valid = pool & lane.keys()
        side_features.append({doc: features(*topics[doc], topic, lane[doc], sparse.get(doc), dense[doc])
                              for doc in sorted(valid)})
    for pair in sorted(scores):
        for left, right in (pair, pair[::-1]):
            if left not in facets[0] or right not in facets[1]:
                continue
            a, b = facets[0][left], facets[1][right]
            if not (a[0] | a[1]).isdisjoint(b[0] | b[1]):
                oriented[left, right] = (side_features[0][left], side_features[1][right])
    return ranks, facets, baseline, oriented, common, perf_counter() - extra_started


def dataset(catalogue, embedding, directory):
    index = linked.replay.live.LocalIndex(catalogue, linked.previous.mixed.QwenTorchEmbedding(embedding), directory)
    words = {doc: tuple(terms(value) for value in text) for doc, text in catalogue.documents.items()}
    return index, words, plans.typed.RolePostings(catalogue)


def fit(index, words, postings, requests, marks):
    samples = {}
    for atoms in requests:
        _, _, _, oriented, _, _ = candidates(index, words, postings, atoms)
        for docs, row in oriented.items():
            for side, (doc, value) in enumerate(zip(docs, row, strict=True)):
                topic = atoms[side][0]
                samples[topic, doc] = (value, topic in marks.get(doc, ()))
    x = np.asarray([values for values, _ in samples.values()], dtype=np.float64)
    y = np.asarray([known for _, known in samples.values()], dtype=np.float64)
    if not 0 < y.sum() < len(y):
        raise ValueError("Require observed positive and unjudged examples")
    x = np.column_stack([np.ones(len(x)), x])
    weights = np.where(y == 1, (0.5 * len(y)) / y.sum(), (0.5 * len(y)) / (len(y) - y.sum()))
    penalty = np.diag([0.] + [1.] * (x.shape[1] - 1))
    coefficients = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty, x.T @ (weights * y))
    return coefficients, dict(examples=len(y), observed_positive=int(y.sum()))


def select(ranks, facets, baseline, oriented, coefficients, arm):
    if arm == "joint" or not oriented:
        return baseline
    weighted, options = plans.triple_plan(ranks, facets, baseline, plans.ALL_EDGES)
    if arm == "rank_set":
        return weighted
    scores = {}
    for pair, features_by_side in oriented.items():
        score = min(coefficients[0] + value @ coefficients[1:]
                    for value in np.asarray(features_by_side))
        key = tuple(sorted(pair))
        scores[key] = max(scores.get(key, -np.inf), score)
    original = tuple(sorted(baseline))

    def priority(documents):
        value = max((scores.get(pair, -np.inf) for pair in combinations(documents, 2)))
        return -value, documents != original, documents
    return min(options, key=priority)


def evaluate(index, words, postings, requests, marks, coefficients, reverse):
    rows, timing, availability = {}, {arm: [] for arm in ARMS}, {arm: 0 for arm in ARMS}
    pool_known = 0
    descriptions = list(enumerate(requests))
    for number, atoms in reversed(descriptions) if reverse else descriptions:
        ranks, facets, baseline, oriented, common_seconds, feature_seconds = candidates(index, words, postings, atoms)

        def tagged(proof):
            return all(topic in marks.get(doc, ()) for (topic, _), doc in zip(atoms, proof[1:], strict=True))
        pool_known += bool(plans.label_supported_pairs(postings, atoms, ranks, marks))
        row = dict(atoms=atoms, arms={}, topic_pair=json.dumps([topic for topic, _ in atoms]))
        order = ARMS[::-1] if bool(number % 2) != reverse else ARMS
        for arm in order:
            begin = perf_counter()
            selection = select(ranks, facets, baseline, oriented, coefficients, arm)
            if len(set(selection)) != 3:
                raise ValueError("Expected exactly three reads")
            loaded = index.open_sources(selection)
            base = plans.live.BASE
            proofs = base.choose_proofs(base.read_articles(SimpleNamespace(ownership=loaded), selection, atoms, ranks), ranks)
            match = proofs[0] if proofs else None
            json.dumps(match)
            extra = feature_seconds if arm == "learned_set" else 0.
            timing[arm].append(1000 * (common_seconds + extra + perf_counter() - begin))
            selected = [ranks[side][doc] for side, doc in enumerate(match[1:])] if match else []
            row["arms"][arm] = dict(match=match, priority=(max(selected), sum(selected)) if selected else None,
                                    known=bool(match) and tagged(match), opened=list(selection))
            availability[arm] += any(tagged(proof) for proof in proofs)
        rows[number] = row
    complete = [rows[i] for i in range(len(rows))]
    return dict(
        requests=len(complete), tag_supported_in_pool=pool_known, reachable_after_reads=availability,
        p95_ms={arm: float(np.quantile(values, .95)) for arm, values in timing.items()},
        comparisons=plans.paired.compare(complete, (("joint", "rank_set"), ("joint", "learned_set"))),
        decision_signature=plans.paired.signature(complete),
    )


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = linked.previous.live.verify_embedding(args.embedding)
    if checkpoint != reference["embedding"]:
        raise ValueError("Changed encoder")
    base = plans.live.BASE
    plos, _ = base.pinned.load_fixture(args.plos_fixture, [args.plos])
    training_marks = base.publisher_topics([args.plos])
    training_requests = base.query_panel(base.annotation_postings(plos, training_marks))
    panels = {}
    with TemporaryDirectory(dir=args.output.parent) as directory:
        work = Path(directory)
        coefficients, receipt = fit(*dataset(plos, args.embedding, work / "plos"), training_requests, training_marks)
        for name in linked.PANELS:
            projection = getattr(args, name)
            if linked.prior.digest(projection) != reference["projection_sha256"][name]:
                raise ValueError("Changed held-out projection")
            catalogue = linked.load(projection, getattr(args, name + "_manifest"))
            marks = {doc: {" ".join(topic.casefold().split()) for topic in row["topics"]}
                     for doc, row in catalogue.ownership.items()}
            requests = base.query_panel(base.annotation_postings(catalogue, marks))
            report = evaluate(*dataset(catalogue, args.embedding, work / name),
                              requests, marks, coefficients, args.reverse)
            expected = reference["panels"][name]["depths"][str(DEPTH)]
            natural = report["comparisons"]["joint_to_rank_set"]["natural"]["arms"]["joint"]
            if (report["requests"], natural["issued"], natural["known"]) != (
                    reference["panels"][name]["query_count"], expected["issued"], expected["tagged"]):
                raise ValueError("Original test plan did not reproduce")
            panels[name] = report
    linked.prior.write_json(args.output, dict(
        panels=panels, fit=receipt | {"coefficients": coefficients.tolist(), "requests": len(training_requests)},
        depth=DEPTH, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        reference_sha256=linked.prior.digest(args.reference), script_sha256=linked.prior.digest(Path(__file__)),
        embedding=checkpoint, reverse=args.reverse,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "plos_fixture", "extended", "extended_manifest", "confirmation", "confirmation_manifest",
                  "embedding", "reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
