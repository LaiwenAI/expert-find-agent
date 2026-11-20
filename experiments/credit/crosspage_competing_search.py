"""Test cheap conjunctive nomination and bounded verification in competing pools."""

import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.calibrated_topic_evidence import margins
from experiments.credit.crosspage_audited_search import STOP, tokens
from experiments.credit.crosspage_condition_search import TASK, assignment, page_index, score_held
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = {
    "whole_dense": None, "whole_reader": None,
    "bm25_raw": ("lexical", 0), "dense_raw": ("dense", 0), "hybrid_raw": ("hybrid", 0),
    "dense_2": ("dense", 2), "dense_4": ("dense", 4),
    "hybrid_2": ("hybrid", 2), "hybrid_4": ("hybrid", 4),
    "bm25_2": ("lexical", 2),
}


def postings_index(texts):
    """Compiled BM25 page weights over the same prefixes available to dense."""
    counts = {doc: [Counter(tokens(page)) for page in pages] for doc, pages in texts.items()}
    frequency = Counter(term for pages in counts.values() for page in pages for term in page)
    length = [sum(page.values()) for pages in counts.values() for page in pages]
    total, mean = len(length), float(np.mean(length))
    postings = {}
    for doc, pages in counts.items():
        raw = defaultdict(list)
        for i, page in enumerate(pages):
            length_norm = 1.2 * (.25 + .75 * sum(page.values()) / mean)
            for term, count in page.items():
                idf = math.log1p((total - frequency[term] + .5) / (frequency[term] + .5))
                raw[term].append((i, idf * (count / (count + length_norm)) * 2.2))
        postings[doc] = {
            term: (np.asarray(ids, dtype=np.int32), np.asarray(weights, dtype=np.float32))
            for term, rows in raw.items() for ids, weights in [zip(*rows)]
        }
    return postings


def sparse_scores(keywords, postings, page_count):
    scores = np.zeros((len(keywords), page_count), dtype=np.float32)
    for facet, terms in enumerate(keywords):
        for term in terms:
            if term in postings:
                page, weight = postings[term]
                scores[facet, page] += weight
    return scores


def ordered(scores):
    return sorted(scores, key=lambda doc: (-scores[doc][0], -scores[doc][1], doc))


def complete_at_partial_caps(rows):
    """Retrospective diagnostic. All requests tied at a margin enter together."""
    result = {"5": 0, "10": 0}
    complete = partial = 0
    curve = sorted(rows, key=lambda row: row["margin"], reverse=True)
    for _, bucket in groupby(curve, key=lambda row: row["margin"]):
        for row in bucket:
            complete += row["rank"] == 1
            partial += row["rank"] != 1
        for cap in result:
            if partial <= int(cap):
                result[cap] = complete
    return result


def select(conditions, doc_ids, texts, embeds, masks, postings, encoder, reader, route, verify):
    """Choose distinct evidence pages, then verify at most the top two or four documents."""
    prompts = [f"The report substantively covers {condition}." for condition in conditions]
    vectors = None if route == "lexical" else encoder.encode(prompts, query=True)
    keywords = [sorted(set(tokens(condition)) - STOP) for condition in conditions]
    scores, chosen = {}, {}
    for doc in doc_ids:
        arity = len(conditions)
        if route != "dense":
            sparse = sparse_scores(keywords, postings[doc], len(texts[doc]))
        if route == "lexical":
            matrix = sparse
        else:
            matrix = vectors @ embeds[doc].T
            if route == "hybrid":
                # Saturating contribution preserves semantic ranking when no
                # term matches and bounds the lexical change to 0.15.
                matrix += .15 * sparse / (sparse + 4)
        if len(masks[doc]) - sum(masks[doc]) >= arity:
            matrix[:, masks[doc]] = -np.inf
        selected = assignment(matrix, True)
        chosen[doc] = selected
        values = [matrix[i, page] for i, page in enumerate(selected)]
        scores[doc] = min(values), sum(values)
    ranked = ordered(scores)
    reads, clipped = 0, 0
    top_margin = scores[ranked[0]][0]
    if verify:
        verified = ranked[:verify]
        pairs = [(prompt, texts[doc][page]) for doc in verified
                 for prompt, page in zip(prompts, chosen[doc], strict=True)]
        values, clipped = margins(reader, pairs, TASK)
        verified_scores = {}
        for i, doc in enumerate(verified):
            group = values[i * len(conditions):(i + 1) * len(conditions)]
            verified_scores[doc] = min(group), sum(group)
        ranked = ordered(verified_scores) + ranked[verify:]
        top_margin = verified_scores[ranked[0]][0]
        reads = len(pairs)
    return ranked, top_margin, chosen, reads, clipped


def evaluate(queries, index, postings, encoder, reader, offset):
    texts, embeds, physical, masks = index
    records = defaultdict(list)
    names = tuple(ARMS)
    for i, query in enumerate([queries[0], *queries]):
        shift = (i + offset) % len(names)
        for arm in names[shift:] + names[:shift]:
            start = perf_counter()
            if ARMS[arm] is None:
                ranked, margin, chosen, clipped, reads = score_held(query, *index, encoder, reader, arm)
            else:
                ranked, margin, chosen, reads, clipped = select(
                    query["conditions"], sorted(query["grades"]), texts, embeds, masks, postings,
                    encoder, reader, *ARMS[arm],
                )
            top = ranked[0]
            pages = [physical[top][page] for page in chosen[top]]
            json.dumps({"document": top, "pages": pages})
            elapsed = (perf_counter() - start) * 1000
            if i:
                target = next(doc for doc, grade in query["grades"].items() if grade == query["arity"])
                records[arm].append(dict(
                    query=query["query_id"], n=query["arity"], rank=ranked.index(target) + 1,
                    document=top, pages=pages, reads=reads, clipped=clipped,
                    margin=float(margin), ms=elapsed,
                ))
    result = {}
    for arm, rows in records.items():
        counts = dict(
            complete_first=sum(row["rank"] == 1 for row in rows),
            complete_top2=sum(row["rank"] <= 2 for row in rows),
            reads=sum(row["reads"] for row in rows), clipped=sum(row["clipped"] for row in rows),
            p50_p95_ms=np.quantile([row["ms"] for row in rows], [.5, .95]).tolist(),
            by_arity={str(n): sum(row["rank"] == 1 and row["n"] == n for row in rows) for n in (2, 3)},
        )
        if rows[0]["reads"]:
            counts["fixed_cutoff"] = {
                str(cutoff): {
                    "complete": sum(row["rank"] == 1 and row["margin"] > cutoff for row in rows),
                    "partial": sum(row["rank"] != 1 and row["margin"] > cutoff for row in rows),
                } for cutoff in (0, -4)
            }
        counts["best_at_observed_partial_cap"] = complete_at_partial_caps(rows)
        result[arm] = counts | {"evidence": [{k: v for k, v in row.items() if k != "ms"} for row in rows]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "manifest", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["panel_sha256"] != digest(args.panel):
        raise ValueError("Panel differs from manifest")
    panel = json.loads(args.panel.read_bytes())
    revision = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    index = page_index(panel["documents"], encoder)
    postings = postings_index(index[0])
    result = evaluate(panel["queries"], index, postings, encoder, reader, args.offset)
    write_json(args.output, dict(
        queries=len(panel["queries"]), documents=len(index[0]),
        pages=sum(map(len, index[0].values())), gpu=torch.cuda.get_device_name(),
        embedding_revision=revision["revision"], reranker_revision=reader.provenance["revision"],
        panel_sha256=manifest["panel_sha256"], script_sha256=digest(Path(__file__)),
        offset=args.offset, results=result,
    ))


if __name__ == "__main__":
    main()
