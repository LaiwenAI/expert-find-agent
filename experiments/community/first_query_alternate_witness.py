"""Allocate one verification batch to new authors or complementary answers."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from experiments.community.distilled_first_query_reads import EvidenceIndex, paired_delta, question_text
from experiments.community.evaluate_crossvalidated_witness import connected_groups
from experiments.community.first_query_owner_routing import best_sources, load_panel
from experiments.community.witness_matched_issuance import cross_site_overlap
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("widen", "dense_alternate", "residual_alternate")
ANCHORS = 3
BUDGET = 5
ALTERNATE_DEPTH = 3


def group(order, owners):
    postings = defaultdict(list)
    for doc in order:
        postings[owners[doc]].append(int(doc))
    return postings


def nominate(index, encoder, text):
    vector = encoder.encode([text], query=True)[0]
    similarities = index.embeddings @ vector
    return np.argsort(-similarities, kind="stable"), similarities


def plan(index, postings, similarities, text, arm):
    heads = [docs[0] for docs in list(postings.values())[:BUDGET]]
    anchor_ids = heads[:ANCHORS]
    qterms = set(tokens(text[:950])) if arm == "residual_alternate" else set()
    gap = {head: qterms - index.terms[head] for head in anchor_ids}

    def priority(doc, head):
        # Favour candidates covering new query terms, then the original dense score.
        new_terms = gap[head].intersection(index.terms[doc])
        residual = sum(index.idf[t] for t in sorted(new_terms))
        return residual, similarities[doc], -doc

    alternatives = []
    for head in anchor_ids:
        tail = postings[index.owners[head]][1:1 + ALTERNATE_DEPTH]
        if tail:
            doc = max(tail, key=lambda i: priority(i, head))
            alternatives.append((priority(doc, head), doc))
    alternatives.sort(reverse=True)
    return (anchor_ids + [doc for _, doc in alternatives] + heads[ANCHORS:])[:BUDGET]


def issue(index, docs, margins):
    best = {}
    for doc, score in zip(docs, margins, strict=True):
        owner = index.owners[doc]
        if score > 0 and (owner not in best or (score, -doc) > (best[owner][0], -best[owner][1])):
            best[owner] = (score, doc)
    return [doc for _, doc in sorted(best.values(), key=lambda row: (-row[0], row[1]))[:BUDGET]]


def audit_shortlist(audit, postings, gold):
    first = list(postings.values())[:10]
    known_heads = {docs[0] for docs in first}.intersection(gold)
    known_first_ten_sources = set().union(*first).intersection(gold)
    if known_heads:
        audit["known_top10_head"] += 1
    elif known_first_ten_sources:
        audit["only_alternate_on_top10_author"] += 1
    else:
        audit["no_known_author_in_top10"] += 1


def evaluate(data, encoder, reader, offset, expected=None):
    index = EvidenceIndex(data, encoder)
    queries = data["queries"]
    records = defaultdict(list)
    shortlist = Counter()
    for i, query in enumerate([queries[0], *queries]):
        text = question_text(query)
        gold = {index.ids[doc] for doc in query["answers"]}
        shift = (i + offset) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            order, similarities = nominate(index, encoder, text)
            if arm == "widen":
                chosen = best_sources(order, index.owners)
            else:
                chosen = plan(index, group(order, index.owners), similarities, text, arm)
            if len(chosen) != BUDGET or len(set(chosen)) != BUDGET:
                raise ValueError("Every policy must inspect exactly five different answers")
            margins, clipped = index.read(text, chosen, reader)
            issued = issue(index, chosen, margins)
            elapsed = (perf_counter() - start) * 1000
            if i:
                records[arm].append((issued, gold, chosen, clipped, elapsed))
                if arm == "widen":
                    audit_shortlist(shortlist, group(order, index.owners), gold)
    summary = {}
    groups = connected_groups(queries)
    baseline = records["widen"]
    for arm, rows in records.items():
        metrics = Counter()
        for (sent, gold, inspected, clipped, _), (_, _, baseline_inspected, *_) in zip(rows, baseline, strict=True):
            metrics.update(
                inspected=BUDGET, distinct_checked_owners=len({index.owners[doc] for doc in inspected}),
                selected_known=len(gold.intersection(inspected)), known_issued=len(gold.intersection(sent)),
                issued=len(sent), clipped=clipped,
                known_newly_selected=len(gold.intersection(inspected).difference(baseline_inspected)),
                known_displaced=len(gold.intersection(baseline_inspected).difference(inspected)),
            )
        summary[arm] = dict(
            **metrics, p50_p95_ms=np.quantile([row[-1] for row in rows], [.50, .95]).tolist(),
            versus_widen=paired_delta(rows, baseline, groups),
            versus_widen_matched_issuance=paired_delta(rows, baseline, groups, matched=True),
        )
    if expected is not None:
        observed = summary["widen"]
        if (observed["known_issued"], observed["issued"], observed["selected_known"]) != expected:
            raise ValueError("Prior fixed five did not reproduce")
    return dict(queries=len(queries), groups=len(set(groups)), shortlist=dict(shortlist), arms=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "unix", "mathoverflow", "embedding", "reranker", "unix_reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    dev, _ = load_panel(args.development)
    unix, unix_sha = load_panel(args.unix)
    overflow, overflow_sha = load_panel(args.mathoverflow)
    overflow, exclusions = cross_site_overlap((dev, unix), overflow)
    record = json.loads(args.unix_reference.read_bytes())["runs"][0]
    if record["held_snapshot_sha256"] != unix_sha or record["cross_site_exclusions"] != 0:
        raise ValueError("Unexpected Unix reference")
    reference = record["arms"]["dense5"]
    expected = tuple(reference[key] for key in ("known_issued", "issued", "known_available"))
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    panels = {
        "unix": dict(snapshot_sha256=unix_sha, **evaluate(unix, encoder, reader, args.order_offset, expected)),
        "mathoverflow": dict(snapshot_sha256=overflow_sha, **evaluate(overflow, encoder, reader, args.order_offset)),
    }
    write_json(args.output, dict(
        panels=panels, mathoverflow_cross_site_exclusions=exclusions,
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.unix_reference),
        order_offset=args.order_offset, gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    main()
