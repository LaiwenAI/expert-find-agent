"""Compare packed role-safe verification and guards on ordered citation output."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("base", "fixed_width", "packed", "packed_broad_guard", "packed_output_guard")


def output_risk(values, owners, tolerance):
    """Check whether interval uncertainty can change any emitted witness."""
    ranked = sorted((pos for pos in values if owners[pos]), key=lambda pos: -values[pos])
    emitted = set()
    for index, pos in enumerate(ranked):
        value = values[pos]
        if value + tolerance <= holdout.CUTOFF:
            return False
        if value - tolerance <= holdout.CUTOFF:
            return True
        if index + 1 < len(ranked) and value - values[ranked[index + 1]] <= 2 * tolerance:
            # Even the final emitting source must stay ahead of unconsumed ones.
            return True
        emitted.update(owners[pos])
        if len(emitted) >= pilot.TOP:
            return False
    return False


def wave(reader, packed, rows, active, width, owners, arm, tolerance):
    if arm == "base":
        ids = list(range(len(rows)))
        scores, slots = reader.score(rows, ids)
        return dict(zip(ids, scores, strict=True)), slots, False
    if arm == "fixed_width":
        scores, slots = reader.score(rows, active, width=width)
        return dict(zip(active, scores, strict=True)), slots, False
    scores, slots = packed([rows[i] for i in active]) if active else ([], 0)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite packed score")
    values = dict(zip(active, scores, strict=True))
    if arm == "packed":
        return values, slots, False
    risk = (batches.margin_risk(scores, tolerance) if arm == "packed_broad_guard"
            else output_risk(values, owners, tolerance))
    if risk:
        exact, extra, _ = wave(reader, packed, rows, active, width, owners, "base", tolerance)
        return exact, slots + extra, True
    return values, slots, False


def prepare(catalog, reader, topic, roles):
    started = perf_counter()
    sources = catalog.rank(topic)[:8]
    encoded = reader.encode(f"Research about {topic}", [catalog.texts[i] for i in sources])
    owners = [catalog.bindings.get((source, roles), ()) for source in sources]
    return sources, encoded, owners, max(len(row["input_ids"]) for row in encoded), (perf_counter()-started)*1000


def calibrate(catalog, reader, packed):
    errors = []
    for topic, _, roles in batches.queries(catalog):
        sources, rows, owners, width, _ = prepare(catalog, reader, topic, roles)
        base, _, _ = wave(reader, packed, rows, [], width, owners, "base", 0)
        for cover in pilot.CACHE:
            active = batches.eligible(catalog, sources, roles, cover)
            packed_values, _, _ = wave(reader, packed, rows, active, width, owners, "packed", 0)
            errors.extend(abs(v - base[pos]) for pos, v in packed_values.items())
    return max(errors), len(errors)


def run(pilot_projection, projection, manifest, embedding_path, reranker_path, output):
    from functools import partial

    import torch
    from transformers import AttentionInterface

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
    from experiments.retrieval.cof_ragged_shared_weights import measure_storage, packed_view

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if (pilot.digest(pilot_projection), pilot.digest(projection)) != (
            frozen["original_projection_sha256"], frozen["projection_sha256"]):
        raise ValueError("Unexpected topic projections")
    torch.backends.cuda.matmul.allow_tf32 = False
    embed = pilot.verify_checkpoint(embedding_path)
    encoder = QwenTorchEmbedding(embedding_path)
    encoder.model.config.use_cache = False
    ranker = Reranker(reranker_path)
    base = ranker.model.model
    if ranker.model.lm_head.bias is not None or set(base.config.layer_types) != {"full_attention"}:
        raise ValueError("Packed scorer requires the pinned bias-free full-context model")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    with torch.inference_mode():
        weights = ranker.model.lm_head.weight[ranker.labels].contiguous()
        shared = packed_view(base, share_weights=True)
    storage = measure_storage(base, shared)
    if storage["extra_parameter_bytes"]:
        raise ValueError("Packed view duplicated model parameters")
    packed = partial(packed_forward, shared, weights)
    reader = batches.BatchReader(ranker)
    development, target = (pilot.Catalog(json.loads(path.read_bytes())["sources"], encoder, ranker)
                           for path in (pilot_projection, projection))
    tolerance, calibration_scores = calibrate(development, reader, packed)
    outcomes = {cover: {arm: [] for arm in ARMS} for cover in pilot.CACHE}
    for index, (topic, _, roles) in enumerate(batches.queries(target)):
        sources, rows, owners, width, common = prepare(target, reader, topic, roles)
        shift = index % len(ARMS)
        covers = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for cover in covers:
            active = batches.eligible(target, sources, roles, cover)
            for arm in ARMS[shift:] + ARMS[:shift]:
                begin = perf_counter()
                values, slots, fallback = wave(reader, packed, rows, active, width, owners, arm, tolerance)
                result = batches.evidence(target, topic, roles, sources, values)
                ms = common + (perf_counter() - begin) * 1000
                outcomes[cover][arm].append(dict(active=active, values=values, slots=slots, rechecked=fallback,
                    result=dict(result, topic=topic, ms=ms, reads=len(values),
                                characters=sum(len(target.texts[sources[pos]]) for pos in values))))
    reports = {}
    for cover, arms in outcomes.items():
        reports[cover] = {}
        for name, rows in arms.items():
            report = batches.summarize(rows, arms["base"], "base")
            report["paired_p95_vs_fixed_width_ci95_ms"] = holdout.compare_latency(
                [r["result"]["ms"] for r in rows], [r["result"]["ms"] for r in arms["fixed_width"]],
                [r["result"]["topic"] for r in rows],
            )
            reports[cover][name] = report
    pilot.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 script_sha256=pilot.digest(Path(__file__)), manifest_sha256=pilot.digest(manifest),
                                 embedding=embed["revision"], reranker=ranker.provenance["revision"],
                                 storage=storage, tolerance=tolerance, calibration_scores=calibration_scores,
                                 holdout=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pilot_projection", "projection", "manifest", "embedding_path", "reranker_path", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
