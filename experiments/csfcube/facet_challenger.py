"""Test whether matching facet-specific views helps a fixed two-slot verifier."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.csfcube.distilled_facet_transfer import LABELS, views
from experiments.csfcube.frozen_confirmation import load
from experiments.csfcube.grouped_calibration import BUDGETS, VIEWS, cross_validate, partition, summarize
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.learned_confirmation import observation
from experiments.litsearch.source_withholding import TASK


ARMS = ("dense_two", "lead_both", "facet_query", "facet_both", "facet_both_batched", "dense_six")


def query_vectors(encoder, texts):
    return torch.from_numpy(encoder.encode([f"Instruct: {TASK}\nQuery: {text}" for text in texts])).cuda()


def dense_pool(full, available, keys, query):
    ordering = torch.argsort(full[available] @ query, descending=True, stable=True)[:6].tolist()
    if len(ordering) != 6:
        raise ValueError("Insufficient candidates")
    return [keys[i] for i in ordering]


def index(encoder, papers, data, questions):
    with (data / "abstracts-csfcube-preds.jsonl").open() as stream:
        records = {int(r["paper_id"]): r for r in map(json.loads, stream)}
    texts = {row["corpusid"]: paper_text(row) for row in papers}
    vectors = torch.from_numpy(encoder.encode(list(texts.values()))).cuda()
    candidate_facets = set((qid, q["facet"]) for q in questions for qid in q["grades"])
    compact = {}
    counts = Counter()
    for doc, facet in sorted(candidate_facets):
        record = records[doc]
        sentences = record["abstract"]
        labels = record["pred_labels"]
        selected = [span for span, label in zip(sentences, labels) if label in LABELS[facet]][:3]
        if not selected:
            counts[facet] += 1
            selected = sentences[:1]
        header = record["title"] + "\n"
        compact[doc, facet] = (header + " ".join(sentences[:len(selected)]), header + " ".join(selected))
    entries = {text: i for i, text in enumerate(dict.fromkeys(text for pair in compact.values() for text in pair))}
    small = torch.from_numpy(encoder.encode(list(entries))).cuda()
    positions = {doc: i for i, doc in enumerate(texts)}
    return texts, vectors, positions, small, {key: tuple(entries[t] for t in pair) for key, pair in compact.items()}, dict(counts)


def shortlists(encoder, question, rendered, view, indexes, offset):
    _, full, positions, compact, ids, _ = indexes
    start = perf_counter()
    n = question["corpusids"] if view == "withheld" else ()
    keys = sorted(question["grades"].keys() - set(n))
    available = [positions[key] for key in keys]
    query = query_vectors(encoder, [rendered["full"]])[0]
    pool = dense_pool(full, available, keys, query)
    elapsed = (perf_counter() - start) * 1000
    selected = dict(dense_two=(pool[:2], elapsed), dense_six=(pool, elapsed))
    for arm in ARMS[offset:] + ARMS[:offset]:
        if arm in ("dense_two", "dense_six"):
            continue
        start = perf_counter()
        style = "matched_lead" if arm == "lead_both" else "facet"
        candidate_pool = pool
        base_ms = elapsed
        if arm == "facet_both_batched":
            full_query, query = query_vectors(encoder, [rendered["full"], rendered[style]])
            candidate_pool = dense_pool(full, available, keys, full_query)
            base_ms = 0.
        else:
            query = query_vectors(encoder, [rendered[style]])[0]
        if arm == "facet_query":
            candidates = full[[positions[doc] for doc in candidate_pool[1:]]]
        else:
            variant = int(arm != "lead_both")
            candidates = compact[[ids[doc, question["facet"]][variant] for doc in candidate_pool[1:]]]
        challenger = int(torch.argmax(candidates @ query).item()) + 1
        selected[arm] = ([candidate_pool[0], candidate_pool[challenger]], base_ms + (perf_counter() - start) * 1000)
    return selected


def observe(encoder, reranker, question, rendered, view, indexes, i):
    offset = (i + int(view == "withheld")) % len(ARMS)
    nominations = shortlists(encoder, question, rendered, view, indexes, offset)
    texts = indexes[0]
    grades = question["grades"]
    arms = {}
    for arm in ARMS[offset:] + ARMS[:offset]:
        selected, retrieval_ms = nominations[arm]
        start = perf_counter()
        scores, truncated = reranker.score(question["query"], texts, selected, task=TASK)
        if truncated or not np.isfinite(scores).all():
            raise ValueError("Invalid verification scores")
        top, _ = observation(selected, scores)
        arms[arm] = dict(grade=grades[top[0]], scored=top, ms=retrieval_ms + (perf_counter() - start) * 1000,
                         retrieval_ms=retrieval_ms, selected=tuple(selected),
                         hit=any(grades[doc] >= 2 for doc in selected))
    return arms


def report(rows, questions):
    groups = [q["query_id"] for q in questions]
    folds = partition([(str(q["query_id"]), q["grades"]) for q in questions])
    base = rows["dense_two"]
    stages = {}
    for arm, records in rows.items():
        stages[arm] = {}
        for view in VIEWS:
            pairs = [(row[view], ref[view]) for row, ref in zip(records, base, strict=True)]
            stages[arm][view] = dict(
                pair_count=sum(len(a["selected"]) for a, _ in pairs), exposure=sum(a["hit"] for a, _ in pairs),
                changed=sum(a["selected"] != b["selected"] for a, b in pairs),
                gained=sum(a["hit"] and not b["hit"] for a, b in pairs),
                lost=sum(b["hit"] and not a["hit"] for a, b in pairs),
                reranked_strong=sum(a["grade"] >= 2 for a, _ in pairs),
                retrieval_p95_ms=float(np.quantile([a["retrieval_ms"] for a, _ in pairs], .95)))
    outcomes = {}
    for budget in BUDGETS:
        by_arm = {arm: cross_validate(records, folds, budget)[0] for arm, records in rows.items()}
        results = outcomes[str(budget)] = {}
        for arm, predictions in by_arm.items():
            flags = [int((a["present"] or 0) >= 2) - int((b["present"] or 0) >= 2)
                     for a, b in zip(predictions, by_arm["dense_two"], strict=True)]
            results[arm] = dict(summary=summarize(predictions, rows[arm]),
                                strong_delta=clustered_paired_interval(flags, groups))
    batch_agreement = {}
    for view in VIEWS:
        paired = [(batch[view], split[view])
                  for batch, split in zip(rows["facet_both_batched"], rows["facet_both"], strict=True)]
        batch_agreement[view] = dict(shortlists=sum(a["selected"] == b["selected"] for a, b in paired),
                                     reranker_score_tuples=sum(a["scored"] == b["scored"] for a, b in paired))
    return dict(stages=stages, grouped=outcomes, batched_vs_separate=batch_agreement)


def run(data, reference, embedding, reranker_path, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    pinned = json.loads(reference.read_bytes())
    papers, questions, hashes = load(data)
    provenance = verify_checkpoint(embedding)
    reranker = Reranker(reranker_path)
    if hashes != pinned["source_hashes"] or provenance != pinned["embedding"] or reranker.provenance != pinned["reranker"]:
        raise ValueError("Dataset or checkpoint differs from pinned CSFCube benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding, batch_size=32)
    encoder.model.config.use_cache = False
    started = perf_counter()
    indexes = index(encoder, papers, data, questions)
    offline_seconds = perf_counter() - started
    rendered, diagnostics = views(data, questions)
    print(f"Prepared {len(questions)} queries in {offline_seconds:.1f} seconds", flush=True)
    observe(encoder, reranker, questions[0], rendered[0], "present", indexes, 0)
    rows = {arm: [] for arm in ARMS}
    for i, (question, content) in enumerate(zip(questions, rendered, strict=True)):
        states = {view: observe(encoder, reranker, question, content, view, indexes, i)
                  for view in VIEWS[i % 2:] + VIEWS[:i % 2]}
        for arm in ARMS:
            rows[arm].append({view: states[view][arm] for view in VIEWS})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            queries=len(questions), query_papers=len({q["query_id"] for q in questions}),
                            offline_seconds=offline_seconds, query_views=diagnostics,
                            compact_unique_vectors=int(indexes[3].shape[0]), document_fallbacks=indexes[-1],
                            code_sha256=digest(Path(__file__)), reference_sha256=digest(reference),
                            **report(rows, questions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "embedding", "reranker_path", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
