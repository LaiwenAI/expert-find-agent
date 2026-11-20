"""Compare equally sized reviewer evidence sets with background similarity correction."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, paired_stats, read_split
from experiments.retrieval.lrbench_source_facets import normalized_title


NEIGHBORS = 10
DEPTH = 3
ARMS = ("dense", "specificity_selection", "specificity", "two_sided")


def largest(scores, k):
    return np.partition(scores, -k, axis=-1)[..., -k:]


def background(encoder, docs, stored):
    # Fit source penalties on publication text only. Exclude the publication
    # itself and other records with the same normalized title.
    queries = encoder.encode(docs, query=True)
    titles = [normalized_title(doc) for doc in docs]
    vocab = {title: i for i, title in enumerate(dict.fromkeys(titles))}
    title_ids = torch.tensor([vocab[title] for title in titles], device="cuda")
    source_bank = torch.from_numpy(stored).cuda()
    pseudo_queries = torch.from_numpy(queries).cuda()
    parts = []
    for start in range(0, len(stored), 256):
        stop = start + 256
        scores = source_bank[start:stop] @ pseudo_queries.T
        scores.masked_fill_(title_ids[start:stop, None] == title_ids[None, :], -torch.inf)
        parts.append(scores.topk(NEIGHBORS, dim=1).values.mean(dim=1).cpu().numpy())
    means = np.concatenate(parts)
    if not np.isfinite(means).all():
        raise ValueError("Insufficient background publications")
    return means


def score_pair(pair, lookup, bank, qmap, penalties, query_background, arm):
    scores, selections = [], []
    for query, history in pair:
        indices = lookup[history]
        raw = bank[indices] @ qmap[query]
        corrected = raw if arm == "dense" else raw - 0.5 * penalties[indices]
        order = np.argsort(-corrected, kind="stable")[:DEPTH]
        values = raw if arm == "specificity_selection" else corrected
        summary = float(values[order].mean())
        if arm == "two_sided":
            summary -= 0.5 * query_background[query]
        scores.append(summary)
        selections.append(tuple(indices[order].tolist()))
    return scores, selections


def run(release, model, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output path on Slurm CUDA")
    for asset, checksum in ASSET_HASHES.items():
        if digest(release / asset) != checksum:
            raise ValueError(f"Changed upstream asset: {asset}")
    cohorts = {split: read_split(release, split) for split in ("pc", "rc")}
    uses = [item for pairs, _ in cohorts.values() for pair in pairs for item in pair]
    candidates = list(dict.fromkeys(doc for _, history in uses for doc in history))
    targets = {normalized_title(query) for query, _ in uses}
    docs = [doc for doc in candidates if normalized_title(doc) not in targets]
    positions = {doc: idx for idx, doc in enumerate(docs)}
    lookup = {history: np.asarray([positions[doc] for doc in history if doc in positions], dtype=np.int32)
              for _, history in uses}
    if any(not len(indices) for indices in lookup.values()):
        raise ValueError("Empty profile")
    spec = importlib.util.spec_from_file_location("rate_evaluator", release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(model)
    encoder = QwenTorchEmbedding(model)
    encoder.model.config.use_cache = False
    encoder.encode([uses[0][0]], query=True)
    started_index = perf_counter()
    bank = encoder.encode(docs)
    encoded_at = perf_counter()
    penalties = background(encoder, docs, bank)
    compiled_at = perf_counter()
    results = {}
    for split, (pairs, reference) in cohorts.items():
        prediction = {"rate": reference, **{arm: [] for arm in ARMS}}
        durations = {arm: [] for arm in ARMS}
        changes = Counter()
        for i, (pair, ref) in enumerate(zip(pairs, reference, strict=True)):
            queries = list(dict.fromkeys(query for query, _ in pair))
            started = perf_counter()
            qmap = dict(zip(queries, encoder.encode(queries, query=True), strict=True))
            encode_ms = (perf_counter() - started) * 1000
            offset = i % len(ARMS)
            selected = {}
            for arm in ARMS[offset:] + ARMS[:offset]:
                started = perf_counter()
                nearby = {}
                if arm == "two_sided":
                    nearby = {query: float(largest(bank @ v, NEIGHBORS).mean()) for query, v in qmap.items()}
                (positive, negative), selected[arm] = score_pair(pair, lookup, bank, qmap, penalties, nearby, arm)
                durations[arm].append(encode_ms + (perf_counter() - started) * 1000)
                prediction[arm].append(dict(ref, positive_score=positive, negative_score=negative))
            for arm in ARMS:
                changes[arm] += selected[arm] != selected["dense"]
        metrics = paired_stats(evaluator, split, prediction)
        baseline = np.array([r["positive_score"] > r["negative_score"] for r in prediction["dense"]], dtype=int)
        ids = [r["anchor_id"] for r in reference]
        unit = "paper anchor" if split == "pc" else "reviewer anchor"
        for arm in ARMS:
            delta = np.array([r["positive_score"] > r["negative_score"] for r in prediction[arm]], dtype=int) - baseline
            metrics[arm].update(
                changed_evidence_pairs=changes[arm],
                better=int((delta > 0).sum()), worse=int((delta < 0).sum()),
                vs_dense=dict(clustered_paired_interval(delta, ids), sampling_unit=unit),
                encode_plus_scoring_p95_ms=float(np.quantile(durations[arm], .95)),
            )
        results[split] = metrics
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                            input_hashes=ASSET_HASHES, embedding=provenance["revision"], publications=len(docs),
                            overlapping_publications_excluded=len(candidates)-len(docs),
                            document_encoding_seconds=encoded_at-started_index,
                            background_extra_seconds=compiled_at-encoded_at,
                            evidence_depth=DEPTH, neighbors=NEIGHBORS, results=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "model", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
