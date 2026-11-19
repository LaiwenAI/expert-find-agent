"""Compare conjunctive and fused candidate recall at a fixed verification budget."""

import argparse
from collections import Counter
import os
from pathlib import Path
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
    CRITERION_CHARS, CUTOFF, EMBED_TASK, K, GuardedReader, ROLES, read_rows, spans,
)
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY, verify


RRF_K = 60
ANCHOR = 2
ARMS = ("whole_openings", "whole_cover", "rrf_openings", "rrf_cover", "joint_openings", "joint_cover")


def reciprocal(scores):
    sorted_scores = np.sort(scores, axis=0)
    ranks = np.column_stack([len(scores) + 1 - np.searchsorted(sorted_scores[:, col], scores[:, col], side="right")
                             for col in range(scores.shape[1])])
    return 1. / (RRF_K + ranks)


def orders(ref, scores):
    by_owner = np.maximum.reduceat(scores[ref.owner_rows], ref.owner_starts, axis=0)
    original = np.argsort(-by_owner[:, 0], kind="stable")
    whole_rank = np.empty(len(original), dtype=int)
    whole_rank[original] = np.arange(len(original))
    calibrated = reciprocal(by_owner[:, 1:])
    mean = calibrated.mean(axis=1)
    fused = np.lexsort((whole_rank, -mean))
    joint = np.lexsort((whole_rank, -mean, -calibrated.min(axis=1)))
    anchors = original[:ANCHOR].tolist()
    def merge(ranked):
        return list(dict.fromkeys([*anchors, *ranked]))[:K]
    return {"whole": original[:K].tolist(), "rrf": merge(fused), "joint": merge(joint)}


def source_pair(rows, strengths, whole):
    """Greedily maximize the sum of best source strength across criteria."""
    first = max(rows, key=lambda row: float(strengths[row].sum()))
    if len(rows) == 1:
        return [first]
    current = strengths[first]
    second = max((row for row in rows if row != first),
                 key=lambda row: float(np.maximum(strengths[row], current).sum()))
    selected = {first, second}
    return [row for row in whole if row in selected]


def assemble(ref, scores, route_owners):
    strengths = reciprocal(scores[:, 1:])
    old_sources = {}
    covers = {}
    for idx in sorted({idx for owners in route_owners.values() for idx in owners}):
        eid = ref.owner_ids[idx]
        original = sorted(ref.owners[eid], key=lambda row: (-float(scores[row, 0]), ref.source_ids[row]))
        old_sources[idx] = original[:2]
        covers[idx] = source_pair(original, strengths, original)
    output = {}
    for arm in ARMS:
        route, style = arm.split("_", 1)
        selection = old_sources if style == "openings" else covers
        output[arm] = [
            dict(id=ref.owner_ids[idx], evidence=[dict(id=ref.source_ids[row]) for row in selection[idx]])
            for idx in route_owners[route]
        ]
    return output


def summarize(rows):
    report = {}
    for arm in ARMS:
        counts = Counter()
        gains = []
        for row in rows:
            ref, current = (row[key] for key in ("whole_openings", arm))
            base_supported, curr_supported = (set(one["supported"]) for one in (ref, current))
            baseline_candidates = set(ref["owners"])
            gained = curr_supported - base_supported
            counts["supported_experts"] += len(curr_supported)
            counts["requests_with_support"] += bool(curr_supported)
            counts["supported_slots"] += current["slots"]
            counts["verify_pairs"] += current["pairs"]
            counts["supported_experts_added"] += len(gained)
            counts["supported_experts_removed"] += len(base_supported - curr_supported)
            counts["new_supported_experts_not_in_original_candidates"] += len(gained - baseline_candidates)
            counts["different_candidates"] += set(current["owners"]) != baseline_candidates
            counts["new_candidate_slots"] += len(set(current["owners"]) - baseline_candidates)
            gains.append(len(curr_supported) - len(base_supported))
        report[arm] = dict(counts) | {"delta_supported_experts_per_request": paired_interval(gains)}
    return report


def run(queries, corpus, models, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a new output on a Slurm GPU")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed development inputs")
    requests, records = read_rows(queries), read_rows(corpus)
    sources = {row["id"]: row for row in records}
    torch.backends.cuda.matmul.allow_tf32 = False
    embedding_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(embedding_path)
    embedding = QwenTorchEmbedding(embedding_path)
    vectors = embedding.encode([row["text"] for row in records])
    ref = ExactOwnedVectorIndex(vectors, list(sources), [[row["expert_id"]] for row in records])
    index = GpuOwnedVectorIndex(ref)
    reader = GuardedReader(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    rows = []
    encode_plus_score_ms = []
    for item in requests:
        text = item["text"]
        clauses = spans(text)
        parsed = reader.evaluate(text, clauses, "qualification_recheck", cutoff=CUTOFF)
        objective = text.strip().splitlines()[0][:CRITERION_CHARS]
        criteria = [objective, *(text[c["start"]:c["end"]] for c, label
                                  in zip(clauses, parsed["labels"], strict=True)
                                  if ROLES[label] == "prior_qualification")]
        if len(criteria) < 2:
            continue
        query_vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
        old_hits = index.search_vector(query_vector, k=K)["hits"]
        start = perf_counter()
        projected = [f"Instruct: {EMBED_TASK}\nQuery: {criterion}" for criterion in criteria]
        # Project the conditions under the same target; retain the complete request as a separate anchor.
        for col in range(1, len(projected)):
            projected[col] += f"\nProject objective: {objective}"
        condition_vectors = embedding.encode(projected)
        with torch.inference_mode():
            q = torch.as_tensor(query_vector, device=index.device, dtype=torch.float32)
            condition_gpu = torch.as_tensor(condition_vectors, device=index.device, dtype=torch.float32)
            base_scores = torch.mv(index.vectors, q)
            condition_scores = torch.mm(index.vectors, condition_gpu.T)
            scores = torch.cat((base_scores[:, None], condition_scores), dim=1).cpu().numpy()
        encode_plus_score_ms.append((perf_counter() - start) * 1000)
        candidates = assemble(ref, scores, orders(ref, scores))
        earlier = [dict(id=h["id"], evidence=[dict(id=e["id"]) for e in h["evidence"][:2]]) for h in old_hits]
        if candidates["whole_openings"] != earlier:
            raise AssertionError("Changed original candidate and source prefix")
        row = {}
        for arm in ARMS:
            hits = candidates[arm]
            measured = verify(hits, sources, criteria, verifier, "two_wave_full")
            if measured["clipped"]:
                raise AssertionError("Unexpected clipped criterion–source comparison")
            result = measured["ranks"]
            row[arm] = dict(owners=[h["id"] for h in hits], supported=[e for e, yes, _ in result if yes],
                            slots=sum(c is not None for _, _, refs in result for c in refs), pairs=measured["pairs"])
        rows.append(row)
        if len(rows) % 5 == 0:
            print(f"Compared {len(rows)} multi-criterion requests", flush=True)
    if len(rows) != 25:
        raise AssertionError("Changed multi-criterion cohort")
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(rows),
        corpus_sha256=SOURCE_INVENTORY, development_sha256=DEVELOPMENT, script_sha256=digest(Path(__file__)),
        model_revisions={"embedding": provenance["revision"], "role_reader": reader.provenance["revision"],
                         "verifier": verifier.provenance["revision"]},
        anchor=ANCHOR, candidate_cap=K, source_cap=2,
        extra_projection_and_scoring_p50_p95_ms=np.quantile(encode_plus_score_ms, [.5, .95]).tolist(),
        report=summarize(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("queries", "corpus", "models", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    run(**vars(parser.parse_args()))
