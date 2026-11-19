"""Transfer two-source facet nomination to released reviewer-fit comparisons."""

import argparse
from collections import Counter
import importlib.util
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_method_query_view import build_view
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, paired_stats, read_split
from experiments.retrieval.lrbench_source_facets import TASKS, normalized_title


ARMS = ("dense_two", "lead_challenger", "method_challenger")
VIEW = dict(lead_challenger="lead_sentences", method_challenger="method_sentences")
DEPTH = 6


def prepare(encoder, inputs):
    uses = [item for pairs, _ in inputs.values() for pair in pairs for item in pair]
    docs = dict.fromkeys(doc for _, history in uses for doc in history)
    identifiers = {doc: i for i, doc in enumerate(docs)}
    profiles = {history: np.asarray([identifiers[doc] for doc in history], dtype=np.int32)
                for _, history in uses}
    lookup = dict(enumerate(docs))
    compact = {(arm, idx): build_view(doc, view)
               for arm, view in VIEW.items() for idx, doc in lookup.items()}
    texts = dict.fromkeys((*docs, *compact.values()))
    text_ids = {text: i for i, text in enumerate(texts)}
    embeddings = encoder.encode(list(texts))
    full = embeddings[[text_ids[doc] for doc in docs]]
    short = {arm: embeddings[[text_ids[compact[arm, i]] for i in lookup]] for arm in VIEW}
    return profiles, lookup, full, short, len(texts)


def run_pair(encoder, verifier, pair, indexes, arm):
    profiles, docs, full, short, _ = indexes
    queries = list(dict.fromkeys(query for query, _ in pair))
    start = perf_counter()
    needed = queries if arm == "dense_two" else queries + [build_view(q, VIEW[arm]) for q in queries]
    vector_rows = encoder.encode(needed, query=True)
    qids = {text: idx for idx, text in enumerate(queries)}
    selected = []
    dense_summary = []
    for query, history in pair:
        qpos = qids[query]
        indices = profiles[history]
        scores = full[indices] @ vector_rows[qpos]
        order = np.argsort(-scores, kind="stable")[:DEPTH]
        choices = indices[order][:2].tolist()
        if arm != "dense_two" and len(order) > 1:
            alternatives = indices[order[1:]]
            short_scores = short[arm][alternatives] @ vector_rows[qpos + len(queries)]
            choices[1] = int(alternatives[np.argmax(short_scores)])
        dense_summary.append(float(scores[order[:3]].mean()))
        selected.append(choices)
    retrieval_end = perf_counter()
    requests = [(query, docs[idx]) for (query, _), chosen in zip(pair, selected, strict=True) for idx in chosen]
    values, clipped = verifier.score_pairs(requests, task=TASKS["general"])
    if clipped or not np.isfinite(values).all():
        raise ValueError("Invalid cross-encoder scores")
    boundary = len(selected[0])
    decisions = (max(values[:boundary]), max(values[boundary:]))
    end = perf_counter()
    return dict(chosen=selected, decisions=decisions, dense_top3=dense_summary,
                pairs=len(requests), retrieval_ms=(retrieval_end-start)*1000, ms=(end-start)*1000)


def summarize(evaluator, split, reference, observations):
    out = {}
    predictions = {"rate": reference}
    for arm, rows in observations.items():
        predictions[arm] = [dict(ref, positive_score=row["decisions"][0], negative_score=row["decisions"][1])
                            for row, ref in zip(rows, reference, strict=True)]
        baseline = observations["dense_two"]
        differences = [int(a["decisions"][0] > a["decisions"][1]) - int(b["decisions"][0] > b["decisions"][1])
                       for a, b in zip(rows, baseline, strict=True)]
        out[arm] = dict(
            ms_p95=float(np.quantile([row["ms"] for row in rows], .95)),
            retrieval_ms_p95=float(np.quantile([row["retrieval_ms"] for row in rows], .95)),
            reader_pairs=sum(row["pairs"] for row in rows),
            changed_selection=sum(a["chosen"] != b["chosen"] for a, b in zip(rows, baseline, strict=True)),
            changed_winner=sum(d != 0 for d in differences),
            wins=sum(d > 0 for d in differences), losses=sum(d < 0 for d in differences),
            accuracy_delta_vs_dense_two=clustered_paired_interval(differences, [r["anchor_id"] for r in reference]))
    predictions["dense_top_three"] = [
        dict(ref, positive_score=row["dense_top3"][0], negative_score=row["dense_top3"][1])
        for row, ref in zip(observations["dense_two"], reference, strict=True)]
    return dict(accuracy=paired_stats(evaluator, split, predictions), live_routes=out)


def run(release, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    for name, expected in ASSET_HASHES.items():
        if digest(release / name) != expected:
            raise ValueError(f"Changed input asset: {name}")
    inputs = {split: read_split(release, split) for split in ("pc", "rc")}
    for pairs, _ in inputs.values():
        for pair in pairs:
            for query, profile in pair:
                if not profile or any(normalized_title(doc) == normalized_title(query) for doc in profile):
                    raise ValueError("Empty or title-overlapping publication history")
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, max_length=512, batch_size=32)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    started = perf_counter()
    indexes = prepare(encoder, inputs)
    offline = perf_counter() - started
    spec = importlib.util.spec_from_file_location("rate_evaluator", release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    result = {}
    for split, (pairs, reference) in inputs.items():
        rows = {arm: [] for arm in ARMS}
        for arm in ARMS:
            run_pair(encoder, verifier, pairs[0], indexes, arm)
        for i, pair in enumerate(pairs):
            offset = i % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                rows[arm].append(run_pair(encoder, verifier, pair, indexes, arm))
            if (i + 1) % 100 == 0:
                print(f"{split}: {i + 1}/{len(pairs)} pairs", flush=True)
        if len({sum(row["pairs"] for row in observations) for observations in rows.values()}) != 1:
            raise ValueError("Unequal model pair counts")
        result[split] = summarize(evaluator, split, reference, rows)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            input_hashes=ASSET_HASHES, script_sha256=digest(Path(__file__)),
                            embedding=provenance["revision"], reranker=verifier.provenance["revision"],
                            offline_seconds=offline, indexed_unique_views=indexes[-1],
                            profile_documents=len(indexes[1]), results=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
