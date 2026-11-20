"""Test same-author source planning on real community questions and answer text."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_byline_depth as byline_join
from experiments.credit import live_joint_source_request as live
from experiments.credit.vector_hybrid_rrf import RankFusion, DEPTH as RRF_DEPTH
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


DEPTH = 64
ARMS = ("topical", "staged", "joint")
WINDOW_CHARS = 1800
PAIR_CAP = 8
VERIFIER_BATCH = 8
TASK = "Decide whether the answer directly addresses the question and gives a useful explanation."


def load_snapshot(folder):
    path = folder / "snapshot.json"
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if digest(path) != manifest["snapshot_sha256"]:
        raise ValueError("Community snapshot changed")
    data = json.loads(path.read_bytes())
    documents = {str(doc): row for doc, row in data["documents"].items()}
    expected = {str(doc) for doc in manifest["candidate_answers"]}
    if documents.keys() != expected:
        raise ValueError("Manifest and documents disagree")
    return data["queries"], documents, digest(path)


def freeze_cases(queries, documents, canonical):
    if canonical:
        rows = {
            row["qid"]: dict(text=row["question"] + "\n" + row["question_text"], known={doc})
            for doc, row in documents.items()
        }
    else:
        rows = {
            row["qid"]: dict(text=row["title"] + "\n" + row["query"], known=set(map(str, row["answers"])))
            for row in queries
        }
    people = {qid: {documents[doc]["owner"] for doc in row["known"]} for qid, row in rows.items()}
    grouped = defaultdict(list)
    for a, b in combinations(sorted(rows), 2):
        if rows[a]["known"] & rows[b]["known"]:
            continue
        common = people[a] & people[b]
        if common:
            grouped[min(common)].append((a, b))

    def key(pair):
        return sha256(f"{pair[0]}:{pair[1]}".encode()).digest()

    cases = sorted({pair for group in grouped.values()
                    for pair in (sorted(group, key=key)[:PAIR_CAP] if canonical else group)})
    used = set()
    independent = set()
    for pair in sorted(cases, key=key):
        if used.isdisjoint(pair):
            independent.add(pair)
            used.update(pair)
    in_use = {query for pair in cases for query in pair}
    return {q: rows[q] for q in sorted(in_use)}, cases, independent, len(grouped)


def build_scores(queries, documents, embed, root):
    ids = sorted(documents, key=int)
    # Only the answer text is searchable and visible to the verifier.
    # Disambiguating identical text in the lexical builder does not
    # include the parent question or any candidate identity.
    lexical_input = {}
    answer_texts = [documents[doc]["answer"] for doc in ids]
    for i, text in enumerate(answer_texts):
        while text in lexical_input:
            text += "\n"
        lexical_input[text] = i
    root.mkdir()
    lexical = build_lexical(lexical_input, root)
    fusion = RankFusion(range(len(ids)), lexical)
    encoded = embed.encode(answer_texts)
    questions = [row["text"] for row in queries.values()]
    dense = embed.encode(questions, query=True) @ encoded.T
    ranked = {}
    for (qid, text), score in zip(zip(queries, questions, strict=True), dense, strict=True):
        order = fusion.rank(score, text, "array_part")
        if len(order) != min(len(ids), RRF_DEPTH):
            raise AssertionError("Incomplete atomic rank list")
        ranked[qid] = {ids[doc]: rank for rank, doc in enumerate(order[:DEPTH], 1)}
    return ranked


def score_pairs(reader, queries, documents, pairs, cache):
    missing = sorted(set(pairs) - cache.keys())
    clipped = 0
    for offset in range(0, len(missing), VERIFIER_BATCH):
        batch = missing[offset:offset + VERIFIER_BATCH]
        scores, truncated = reader.score_pairs(
            [(queries[q]["text"], documents[d]["answer"][:WINDOW_CHARS]) for q, d in batch],
            task=TASK,
        )
        if len(scores) != len(batch) or not np.isfinite(scores).all():
            raise ValueError("Missing or nonfinite verifier margin")
        cache.update(zip(batch, scores, strict=True))
        clipped += truncated
    return clipped


def paired_sources(ranks, plan, owners):
    return [(a, b) for a, b in product(plan, repeat=2)
            if a != b and a in ranks[0] and b in ranks[1] and owners[a] == owners[b]]


def stage_extra(extra, anchors, pair, ranks, owners, margins):
    def advance(doc):
        return sum(
            1 / ranks[i][anchor]
            for i, qid in enumerate(pair) for anchor in anchors
            if anchor in ranks[i] and margins[qid, anchor] > 0
            and doc in ranks[1 - i] and owners[doc] == owners[anchor]
            and not any(
                other != anchor and other in ranks[1 - i] and owners[other] == owners[anchor]
                and margins[pair[1 - i], other] > 0 for other in anchors
            )
        )
    return max(extra, key=advance)


def choose(pair, ranks, proposals, scores):
    def order(proof):
        a, b = proof
        left, right = scores[pair[0], a], scores[pair[1], b]
        return -min(left, right), -(left + right), byline_join.pair_key(proof, ranks)

    best = min(proposals, key=order, default=None)
    if best is None:
        return None, None
    weak = min(scores[q, d] for q, d in zip(pair, best, strict=True))
    return (best, float(weak)) if weak > 0 else (None, None)


def evaluate(folder, canonical, encoder, reader, root, offset):
    raw_queries, documents, signature = load_snapshot(folder)
    queries, cases, independent, organizing = freeze_cases(raw_queries, documents, canonical)
    rankings = build_scores(queries, documents, encoder, root)
    owners = {doc: row["owner"] for doc, row in documents.items()}
    byline = {doc: {owner} for doc, owner in owners.items()}
    initial, requests = set(), []
    for a, b in cases:
        pair = (a, b)
        ranks = (rankings[a], rankings[b])
        anchors, extras = live.simple_order(ranks, depth=DEPTH)
        topical = [*anchors, extras[0]]
        joint = byline_join.posting_plan(byline, set().union(*ranks), ranks, topical, DEPTH)
        for doc in set(topical + joint):
            initial.update((qid, doc) for qid in pair)
        requests.append((pair, ranks, anchors, extras, topical, joint))
    margins = {}
    clipped = score_pairs(reader, queries, documents, initial, margins)
    needed, plans = set(), []
    for pair, ranks, anchors, extras, topical, joint in requests:
        staged = [*anchors, stage_extra(extras, anchors, pair, ranks, owners, margins)]
        needed.update((qid, staged[-1]) for qid in pair)
        plans.append(dict(topical=topical, joint=joint, staged=staged))
    clipped += score_pairs(reader, queries, documents, needed, margins)

    result = {arm: Counter() for arm in ARMS}
    disjoint = {arm: Counter() for arm in ARMS}
    natural = {arm: [] for arm in ARMS}
    headroom = 0
    for n, ((pair, ranks, _, _, _, _), docs) in enumerate(zip(requests, plans, strict=True)):
        labels = [queries[q]["known"] for q in pair]

        def known(proof):
            return proof is not None and all(doc in target for doc, target in zip(proof, labels, strict=True))

        gold_pairs = ((a, b) for a, b in product(*labels) if a != b and owners[a] == owners[b])
        headroom += any(a in ranks[0] and b in ranks[1] for a, b in gold_pairs)
        arm_offset = (offset + n) % len(ARMS)
        for arm in ARMS[arm_offset:] + ARMS[:arm_offset]:
            sources = docs[arm]
            if len(set(sources)) != 3:
                raise ValueError("Expected exactly three distinct source reads")
            proposals = paired_sources(ranks, sources, owners)
            selected, confidence = choose(pair, ranks, proposals, margins)
            outcome = dict(source_reads=3, logical_verifier_pairs=6,
                           known_available=int(any(map(known, proposals))),
                           issued=int(selected is not None), known=int(known(selected)),
                           unresolved=int(selected is not None and not known(selected)))
            result[arm].update(**outcome)
            if pair in independent:
                disjoint[arm].update(**outcome)
            if confidence is not None:
                natural[arm].append((-confidence, pair, outcome["known"]))
    quota = min(map(len, natural.values()))
    matched = {arm: sum(item[2] for item in sorted(rows)[:quota]) for arm, rows in natural.items()}
    return dict(
        pool_sources=len(documents), paired_requests=len(cases), atomic_questions=len(queries),
        organizing_authors=organizing, disjoint_question_pairs=len(independent),
        known_pair_reachable=headroom, distinct_verifier_pairs=len(margins), clipped_pairs=clipped,
        natural=result, disjoint=disjoint, matched_issuance=quota, matched_known=matched,
        snapshot_sha256=signature,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stats2024", "stats2025", "mathoverflow", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    checkpoint = live.verify_embedding(args.embedding)
    embed, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        results = {}
        for name, folder, canonical in (
            ("stats2024", args.stats2024, False),
            ("stats2025", args.stats2025, False),
            ("mathoverflow_original", args.mathoverflow, True),
        ):
            results[name] = evaluate(folder, canonical, embed, reader, Path(scratch) / name, args.order_offset)
            print(f"{name}: {results[name]['paired_requests']} paired requests", flush=True)
    write_json(args.output, dict(
        depth=DEPTH, answer_window_characters=WINDOW_CHARS, order_offset=args.order_offset,
        gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        reranker_checkpoint=reader.provenance["revision"], script_sha256=digest(Path(__file__)),
        panels=results,
    ))


if __name__ == "__main__":
    main()
