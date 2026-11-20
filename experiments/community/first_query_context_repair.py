"""Isolate evidence-window failures on a fixed first-query shortlist."""

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
from experiments.community.first_query_owner_routing import load_panel
from experiments.community.held_window_routing import EXPLICIT
from experiments.community.witness_matched_issuance import cross_site_overlap
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("lead", "targeted", "split", "full")
CAP = 850
HEAD = 420
BREAK = "\n[…]\n"
TAIL = CAP - HEAD - len(BREAK)


def lexical_weights(docs):
    df = Counter(term for doc in docs for term in set(tokens(doc)))
    return {term: np.log((1 + len(docs)) / (1 + frequency)) for term, frequency in df.items()}


def choose_view(body, query_terms, weights, arm):
    lead = body[:CAP]
    if arm == "lead" or len(body) <= CAP:
        return lead
    if arm == "full":
        return body

    def value(view, terms):
        return sum(weights.get(term, 0) for term in sorted(terms.intersection(tokens(view))))

    start, length = (0, CAP) if arm == "targeted" else (HEAD, TAIL)
    terms = query_terms if arm == "targeted" else query_terms.difference(tokens(body[:HEAD]))
    last = len(body) - length
    offsets = {*range(start, last + 1, length // 2), last}
    best = max(offsets, key=lambda offset: (value(body[offset:offset + length], terms), -offset))
    span = body[best:best + length]
    view = span if arm == "targeted" else body[:HEAD] + BREAK + span
    return view if value(view, query_terms) > value(lead, query_terms) else lead


def verify(reader, questions, views):
    values, clipped = reader.score_pairs(list(zip(questions, views, strict=True)), task=EXPLICIT)
    if len(values) != len(views) or not np.isfinite(values).all():
        raise ValueError("Incomplete verifier output")
    return values, clipped


def evaluate(data, encoder, reader, offset, expected=None):
    index = EvidenceIndex(data, encoder)
    weights = lexical_weights(index.docs)
    queries = data["queries"]
    canon = {index.ids[int(doc)]: f"{row['question']}\n{row['question_text']}"
             for doc, row in data["documents"].items()}
    records = defaultdict(list)
    changes = Counter()
    canonical_counts = Counter()
    rejection = Counter()
    for i, query in enumerate([queries[0], *queries]):
        text = question_text(query)
        request = text[:950]
        qterms = set(tokens(request))
        gold = {index.ids[doc] for doc in query["answers"]}
        shift = (i + offset) % len(ARMS)
        checked, lead_scores = None, None
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            candidates, _ = index.nominate(text, encoder)
            selected = candidates[:5]
            views = [choose_view(index.docs[doc], qterms, weights, arm) for doc in selected]
            margins, clipped = verify(reader, [request] * len(selected), views)
            passed = [selected[j] for j in np.argsort(-np.asarray(margins), kind="stable") if margins[j] > 0]
            elapsed = (perf_counter() - start) * 1000
            if checked is not None and checked != selected:
                raise ValueError("Verifier arms must inspect the same candidate answers")
            checked = selected
            if arm == "lead":
                lead_scores = dict(zip(selected, margins, strict=True))
            if i:
                records[arm].append((passed, gold, selected, clipped, elapsed))
                changes[arm] += sum(view != index.windows[doc] for doc, view in zip(selected, views, strict=True))
                rejection[f"{arm}_available_queries"] += bool(gold.intersection(selected))
                rejection[f"{arm}_recovered_queries"] += bool(gold.intersection(passed))
        if i:
            known = [doc for doc in checked if doc in gold]
            if known:
                original, clipped = verify(reader, [canon[doc][:950] if doc in gold else request for doc in checked],
                                           [index.windows[doc] for doc in checked])
                canonical_counts["clipped"] += clipped
                for doc, margin in zip(checked, original, strict=True):
                    if doc in gold:
                        left = "pass" if lead_scores[doc] > 0 else "fail"
                        right = "pass" if margin > 0 else "fail"
                        canonical_counts[f"query_{left}_canonical_{right}"] += 1
    summaries = {}
    groups = connected_groups(queries)
    baseline = records["lead"]
    for arm, rows in records.items():
        summaries[arm] = dict(
            pairs=5 * len(rows), changed_views=changes[arm], clipped=sum(row[-2] for row in rows),
            known=sum(len(gold.intersection(issued)) for issued, gold, *_ in rows),
            issued=sum(len(issued) for issued, *_ in rows),
            known_available=sum(len(gold.intersection(checked)) for _, gold, checked, *_ in rows),
            p50_p95_ms=np.quantile([row[-1] for row in rows], [.50, .95]).tolist(),
            versus_lead=paired_delta(rows, baseline, groups),
            versus_lead_matched_issuance=paired_delta(rows, baseline, groups, matched=True),
        )
    lead = summaries["lead"]
    if expected is not None and (lead["known"], lead["issued"], lead["known_available"]) != expected:
        raise ValueError("Frozen widening baseline did not reproduce")
    return dict(queries=len(queries), clusters=len(set(groups)), arms=summaries,
                query_outcomes=dict(rejection), canonical_diagnostic=dict(canonical_counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unix", "mathoverflow", "development", "embedding", "reranker", "unix_reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    unix, unix_sha = load_panel(args.unix)
    overflow, overflow_sha = load_panel(args.mathoverflow)
    dev, _ = load_panel(args.development)
    overflow, exclusions = cross_site_overlap((dev, unix), overflow)
    reference = json.loads(args.unix_reference.read_bytes())["runs"][0]["panels"]["unix"]
    if unix_sha != reference["snapshot_sha256"]:
        raise ValueError("Unexpected Unix snapshot")
    expected = tuple(reference["arms"]["widen"][key] for key in ("known_issued", "issued", "selected_known"))
    provenance = verify_checkpoint(args.embedding)
    embed = QwenTorchEmbedding(args.embedding)
    embed.model.config.use_cache = False
    reader = Reranker(args.reranker)
    panels = {
        "unix": dict(snapshot_sha256=unix_sha, **evaluate(unix, embed, reader, args.order_offset, expected)),
        "mathoverflow": dict(snapshot_sha256=overflow_sha, **evaluate(overflow, embed, reader, args.order_offset)),
    }
    write_json(args.output, dict(
        panels=panels, mathoverflow_cross_site_exclusions=exclusions, order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.unix_reference),
        gpu=torch.cuda.get_device_name(), embedding_revision=provenance["revision"],
        reranker_revision=reader.provenance["revision"],
    ))


if __name__ == "__main__":
    main()
