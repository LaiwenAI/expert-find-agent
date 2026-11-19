"""Measure criterion-gap verification with fixed enterprise candidates and sources."""

import argparse
from collections import Counter
from pathlib import Path
import os
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_first_query import (
    CRITERION_CHARS, CUTOFF, EMBED_TASK, K, SOURCE_CHARS, SOURCE_DEPTH, TASK, VERIFY_BATCH,
    GuardedReader, ROLES, read_rows, spans,
)


ARMS = ("full_batch", "two_wave_full", "gap_only")
BUDGETS = (6, 8)
DEVELOPMENT = "30661871ac08a61e7e74d67d75491878c7e4194f217483b60d06426c4b36e4c5"
SOURCE_INVENTORY = "c9e8d6e13bb05900427ab1d785d6024256f61e8bb34f220c3816fb79d1921780"


def tasks(hits, sources, criteria):
    waves = [[] for _ in range(SOURCE_DEPTH)]
    for hit in hits:
        owner = hit["id"]
        for depth, source in enumerate(hit["evidence"][:SOURCE_DEPTH]):
            row = sources[source["id"]]
            if row["expert_id"] != owner:
                raise ValueError("Candidate source ownership changed")
            excerpt = row["text"][:SOURCE_CHARS]
            if excerpt:
                citation = (source["id"], 0, len(excerpt))
                for i, criterion in enumerate(criteria):
                    if len(criterion) <= CRITERION_CHARS:
                        waves[depth].append((owner, i, criterion, excerpt, citation))
    return waves


def score(pairs, verifier):
    positives = {}
    clipped = calls = 0
    for offset in range(0, len(pairs), VERIFY_BATCH):
        batch = pairs[offset:offset + VERIFY_BATCH]
        margins, truncated = verifier.score_pairs([(p[2], p[3]) for p in batch], task=TASK)
        calls += 1
        clipped += truncated
        if truncated:
            continue
        for (owner, condition, _, _, citation), margin in zip(batch, margins, strict=True):
            if not np.isfinite(margin):
                raise ValueError("Nonfinite support margin")
            if margin > 0:
                positives.setdefault((owner, condition), citation)
    return positives, calls, clipped


def ranked(owners, conditions, witness):
    rows = []
    for owner in owners:
        citations = tuple(witness.get((owner, condition)) for condition in range(conditions))
        rows.append((owner, all(citations), citations))
    return sorted(rows, key=lambda row: not row[1])


def verify(hits, sources, criteria, verifier, arm):
    start = perf_counter()
    first_pairs, second_pairs = tasks(hits, sources, criteria)
    if arm == "full_batch":
        count = len(first_pairs) + len(second_pairs)
        witness, calls, clipped = score(first_pairs + second_pairs, verifier)
        first = second = None
    else:
        first, first_calls, first_clipped = score(first_pairs, verifier)
        if arm == "gap_only":
            second_pairs = [pair for pair in second_pairs if (pair[0], pair[1]) not in first]
        count = len(first_pairs) + len(second_pairs)
        second, second_calls, second_clipped = score(second_pairs, verifier)
        witness = second | first
        calls = first_calls + second_calls
        clipped = first_clipped + second_clipped
    owners = [hit["id"] for hit in hits]
    result = ranked(owners, len(criteria), witness)
    return dict(ranks=result, pairs=count, calls=calls, clipped=clipped,
                ms=(perf_counter() - start) * 1000, first=first, second=second)


def replay_budgets(hits, num_criteria, two_wave):
    owners = [hit["id"] for hit in hits]
    first, second = two_wave["first"], two_wave["second"]
    first_sources = sum(bool(hit["evidence"]) for hit in hits)
    available = [hit["id"] for hit in hits if len(hit["evidence"]) >= 2]
    gaps = {owner: sum((owner, i) not in first for i in range(num_criteria)) for owner in available}
    by_gap = sorted((owner for owner in available if gaps[owner]), key=gaps.__getitem__)
    out = {}
    reference = {owner for owner, supported, _ in two_wave["ranks"] if supported}
    for budget in BUDGETS:
        out[budget] = {}
        for policy, order in (("rank_order", available), ("smallest_gap", by_gap)):
            chosen = set(order[:max(0, budget - first_sources)])
            support = second | first
            support = {key: val for key, val in support.items() if key in first or key[0] in chosen}
            selected = {owner for owner, complete, _ in ranked(owners, num_criteria, support) if complete}
            if not selected <= reference:
                raise AssertionError("Budget replay diverged from fixed scores")
            out[budget][policy] = dict(retained=len(selected), reads=first_sources + len(chosen))
    return out


def summarize(rows):
    result = {}
    for arm in ARMS:
        measures = [row["arms"][arm] for row in rows]
        differences = Counter()
        for row, measure in zip(rows, measures, strict=True):
            target = row["arms"]["two_wave_full"]["ranks"]
            actual = measure["ranks"]
            positives = {owner for owner, yes, _ in actual if yes}
            baseline = {owner for owner, yes, _ in target if yes}
            differences["positive_experts"] += len(positives)
            differences["queries_changed_supported_set"] += positives != baseline
            differences["added_positive_experts"] += len(positives - baseline)
            differences["missing_positive_experts"] += len(baseline - positives)
            differences["queries_changed_ranking_or_citations"] += target != actual
        result[arm] = dict(
            **differences,
            verify_p50_p95_ms=np.quantile([m["ms"] for m in measures], [.5, .95]).tolist(),
            factored_complete_p50_p95_ms=np.quantile(
                [row["common_ms"] + m["ms"] for row, m in zip(rows, measures, strict=True)], [.5, .95]).tolist(),
            **{name: sum(m[name] for m in measures) for name in ("pairs", "calls", "clipped")},
        )
    result["budget_replay"] = {}
    for b in BUDGETS:
        policies = ("rank_order", "smallest_gap")
        result["budget_replay"][b] = {
            p: {metric: sum(row["replay"][b][p][metric] for row in rows)
                for metric in ("retained", "reads")} for p in policies
        }
        result["budget_replay"][b]["paired_retention_difference"] = paired_interval([
            row["replay"][b]["smallest_gap"]["retained"] - row["replay"][b]["rank_order"]["retained"] for row in rows])
    return result


def run(queries, corpus, models, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Run on a Slurm GPU with a new output path")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed development input")
    requests, records = read_rows(queries), read_rows(corpus)
    sources = {r["id"]: r for r in records}
    torch.backends.cuda.matmul.allow_tf32 = False
    embedding_path = models / "qwen3-embedding-0.6b"
    embedding_provenance = verify_checkpoint(embedding_path)
    embedding = QwenTorchEmbedding(embedding_path)
    vectors = embedding.encode([r["text"] for r in records])
    reference = ExactOwnedVectorIndex(vectors, list(sources), [[r["expert_id"]] for r in records])
    index = GpuOwnedVectorIndex(reference)
    print(f"Indexed {len(sources)} sources", flush=True)
    reader = GuardedReader(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    rows = []
    warmup = "Find an expert in optical inspection. Prior experience testing polymer films is required."
    orders = tuple(ARMS[i:] + ARMS[:i] for i in range(len(ARMS)))
    for i, text in enumerate([warmup] + [r["text"] for r in requests]):
        begin = perf_counter()
        clauses = spans(text)
        parsed = reader.evaluate(text, clauses, "qualification_recheck", cutoff=CUTOFF)
        qualification = [text[c["start"]:c["end"]] for c, label in zip(clauses, parsed["labels"], strict=True)
                         if ROLES[label] == "prior_qualification"]
        criteria = [text.strip().splitlines()[0][:CRITERION_CHARS], *qualification]
        vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
        hits = index.search_vector(vector, k=K)["hits"]
        common_ms = (perf_counter() - begin) * 1000
        arms = {a: verify(hits, sources, criteria, verifier, a) for a in orders[i % len(orders)]}
        if i:
            rows.append(dict(criteria=len(criteria), common_ms=common_ms, arms=arms,
                             replay=replay_budgets(hits, len(criteria), arms["two_wave_full"])))
        if i and i % 10 == 0:
            print(f"Completed {i} requests", flush=True)
    grouped = {"single_criterion": [r for r in rows if r["criteria"] == 1],
               "multiple_criteria": [r for r in rows if r["criteria"] > 1]}
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), cohort=len(rows),
        development_sha256=DEVELOPMENT, corpus_sha256=SOURCE_INVENTORY,
        script_sha256=digest(Path(__file__)), sources=len(sources), experts=len(reference.owners),
        model_revisions={"embedding": embedding_provenance["revision"], "role_reader": reader.provenance["revision"],
                         "verifier": verifier.provenance["revision"]},
        report=summarize(rows),
        strata={name: {"requests": len(group), "report": summarize(group)} for name, group in grouped.items() if group},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("queries", "corpus", "models", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(**vars(parser.parse_args()))
