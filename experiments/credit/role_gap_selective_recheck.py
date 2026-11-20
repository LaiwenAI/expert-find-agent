"""Recheck only rank- or acceptance-ambiguous evidence within a packed batch."""

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import packed_topic_guard as guard
from experiments.credit import role_field_ranking as field
from experiments.credit import role_gap_title_refill as gap
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("padded", "fixed_width", "packed", "full_recheck", "selective_recheck")


def ambiguous_rows(values, owners, tolerance):
    ranks = sorted(
        (row for row, value in values.items() if owners[row] and value + tolerance > holdout.CUTOFF),
        key=lambda row: -values[row],
    )
    uncertain = {row for row in ranks if values[row] - tolerance <= holdout.CUTOFF}
    for left, right in zip(ranks, ranks[1:]):
        if values[left] - values[right] <= 2 * tolerance:
            uncertain.update((left, right))
    return sorted(uncertain)


def execute(reader, packed, encoded, active, width, owners, arm, tolerance):
    if arm == "padded":
        ids = range(len(encoded))
        scores, tokens = reader.score(encoded, ids)
        return dict(zip(ids, scores, strict=True)), {}, tokens, False, {}, []
    if arm == "fixed_width":
        scores, tokens = reader.score(encoded, active, width=width)
        return dict(zip(active, scores, strict=True)), {}, tokens, False, {}, []
    scores, tokens = packed([encoded[i] for i in active]) if active else ([], 0)
    initial = dict(zip(active, scores, strict=True))
    if arm == "packed" or not guard.output_risk(initial, owners, tolerance):
        return initial, initial, tokens, False, {}, []
    ids = range(len(encoded)) if arm == "full_recheck" else ambiguous_rows(initial, owners, tolerance)
    values, extra = reader.score(encoded, ids, width=width)
    rechecked = dict(zip(ids, values, strict=True))
    escaped = [i for i, score in rechecked.items() if i in initial and abs(score - initial[i]) > tolerance]
    if escaped and arm == "selective_recheck":
        full = range(len(encoded))
        values, fallback_tokens = reader.score(encoded, full)
        return dict(zip(full, values, strict=True)), initial, tokens + extra + fallback_tokens, True, rechecked, escaped
    return initial | rechecked, initial, tokens + extra, arm == "full_recheck", rechecked, escaped


def serialize(catalog, topic, pair, sources, scores):
    witnesses = batches.evidence(catalog, topic, pair, sources, scores)
    response = json.dumps(witnesses["witnesses"], ensure_ascii=False, separators=(",", ":")).encode()
    return witnesses, response


def run_panel(catalog, titles, min_support, reader, packed, tolerance, reference):
    topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= min_support)
    queries = list(pilot.query_panel(catalog, topics))
    results = {mask: {arm: [] for arm in ARMS} for mask in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        for mask in (pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))):
            start = perf_counter()
            vector = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
            ranked = catalog.rank_for_vector(topic, vector)
            excluded = catalog.visible[mask] - catalog.person[pair]
            sources = gap.refill(ranked, lambda: np.argsort(-(titles @ vector), kind="stable").tolist(), excluded)
            encoded = reader.encode(f"Research about {topic}", [catalog.texts[i] for i in sources])
            owners = [catalog.bindings.get((source, pair), ()) for source in sources]
            active = [i for i, source in enumerate(sources) if source not in excluded]
            width = max(len(row["input_ids"]) for row in encoded)
            common = (perf_counter() - start) * 1000
            rotate = index % len(ARMS)
            observed = {}
            for arm in ARMS[rotate:] + ARMS[:rotate]:
                start = perf_counter()
                values, packed_values, tokens, full, rechecked, escaped = execute(
                    reader, packed, encoded, active, width, owners, arm, tolerance,
                )
                witnesses, response = serialize(catalog, topic, pair, sources, values)
                observed[arm] = dict(
                    values=values, packed_values=packed_values, rechecked=rechecked, response=response,
                    witnesses=witnesses, tokens=tokens, full=full, escaped=len(escaped),
                    ms=common + (perf_counter() - start) * 1000, topic=topic,
                )
            for arm in ARMS:
                base = observed["padded"]
                row = observed[arm]
                row["counters"] = Counter(
                    exact_responses=row["response"] == base["response"], executed_token_slots=row["tokens"],
                    full_rechecks=row["full"], requests_rechecked=bool(row["rechecked"]),
                    rows_rechecked=len(row["rechecked"]), escaped_rechecks=row["escaped"],
                    changed_recheck_scores=sum(value != base["values"][i] for i, value in row["rechecked"].items()),
                    raw_packed_scores_outside_envelope=sum(
                        abs(value - base["values"][i]) > tolerance for i, value in row["packed_values"].items()
                    ),
                )
                results[mask][arm].append(row)
    summary = {}
    for mask, arms in results.items():
        base = arms["padded"]
        prior = reference[str(mask)]["gap_title_four"]["compared"]["topic_four"]
        for key in ("grounded", "issued", "cited_off_label"):
            if sum(row["witnesses"][key] for row in base) != prior[key]:
                raise ValueError(f"Padded baseline did not reproduce the frozen {key} count")
        summary[mask] = {}
        for arm, rows in arms.items():
            times = [row["ms"] for row in rows]
            summary[mask][arm] = dict(
                **sum((row["counters"] for row in rows), Counter()),
                requests=len(rows), p50_p95_p99_ms=np.quantile(times, [.50, .95, .99]).tolist(),
                paired_p95_vs_fixed_width_ci95_ms=holdout.compare_latency(
                    times, [row["ms"] for row in arms["fixed_width"]], [row["topic"] for row in rows],
                ),
            )
    return summary


def run(plos, elife, mesh_snapshot, gap_reference, calibration_reference, embedding, reranker, output):
    import torch
    from transformers import AttentionInterface

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
    from experiments.retrieval.cof_ragged_shared_weights import measure_storage, packed_view

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    # The public projections and checkpoints are the same as in the fixed-budget gap replay.
    targets = json.loads(gap_reference.read_bytes())["run"]
    frozen = json.loads(mesh_snapshot.read_bytes())
    tolerance = json.loads(calibration_reference.read_bytes())["tolerance"]
    if tolerance != .4375 or pilot.digest(elife) != frozen["projection_sha256"]:
        raise ValueError("Expected the frozen numerical envelope and eLife projection")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    rerank = Reranker(reranker)
    if (encoder_info["revision"], rerank.provenance["revision"]) != (targets["embedding"], targets["reranker"]):
        raise ValueError("Expected the frozen checkpoints")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    backbone = rerank.model.model
    if rerank.model.lm_head.bias is not None or set(backbone.config.layer_types) != {"full_attention"}:
        raise ValueError("Expected a bias-free full-attention model")
    with torch.inference_mode():
        shared = packed_view(backbone, share_weights=True)
        weights = rerank.model.lm_head.weight[rerank.labels].contiguous()
    if measure_storage(backbone, shared)["extra_parameter_bytes"]:
        raise ValueError("Packed backbone must share the padded parameters")
    packed = partial(packed_forward, shared, weights)
    reader = batches.BatchReader(rerank)
    reports = {}
    for name, source, support in (("plos", plos, 4), ("elife", elife, scope.MINIMUM_SUPPORT)):
        catalog, titles, _ = field.data_model(source, encoder, rerank)
        if name == "elife":
            complete = {i for i, article in enumerate(catalog.articles) if article["declaration_complete"]}
            catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
            mesh.assign_topics(catalog, frozen["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
            for term in tuple(catalog.topics):
                if not scope.MINIMUM_SUPPORT <= len(catalog.topics[term]) <= scope.MAXIMUM_SUPPORT:
                    del catalog.topics[term]
        warm = sorted(catalog.topics)[0]
        warm_rows = reader.encode(f"Research about {warm}", catalog.texts[:4])
        reader.score(warm_rows, range(4))
        packed(warm_rows)
        reports[name] = run_panel(catalog, titles, support, reader, packed, tolerance, targets[name]["metrics"])
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=pilot.digest(Path(__file__)),
        gap_reference_sha256=pilot.digest(gap_reference), calibration_reference_sha256=pilot.digest(calibration_reference),
        plos_projection_sha256=pilot.digest(plos), elife_projection_sha256=pilot.digest(elife),
        tolerance=tolerance, reports=reports,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "elife", "mesh_snapshot", "gap_reference", "calibration_reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
