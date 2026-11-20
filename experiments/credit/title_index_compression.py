"""Measure compact title vectors through nomination and expert citation output."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import mesh_stage_diagnostic as diagnosis
from experiments.credit import role_field_ranking as field
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


FORMATS = ("fp32", "int8", "sign1")
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def quantize(values):
    peak = np.max(np.abs(values), axis=-1, keepdims=True)
    scale = np.maximum(peak / 127, np.finfo(np.float32).tiny)
    return np.rint(values / scale).astype(np.int8), scale.astype(np.float32)


def packed_sign(values):
    return np.packbits(values >= 0, axis=-1)


def build(title):
    codes, scales = quantize(title)
    return {"fp32": (title,), "int8": (codes, scales[:, 0]), "sign1": (packed_sign(title),)}


def rank_title(arm, index, query):
    if arm == "fp32":
        scores = index[0] @ query
    elif arm == "int8":
        query_codes, query_scale = quantize(query)
        codes, scales = index
        # NumPy's general integer matmul may allocate int32 temporaries.
        scores = np.matmul(codes, query_codes, dtype=np.int32) * scales * query_scale[0]
    else:
        hamming = POPCOUNT[np.bitwise_xor(index[0], packed_sign(query))].sum(axis=-1)
        return np.argsort(hamming, kind="stable").tolist()
    return np.argsort(-scores, kind="stable").tolist()


def collect(catalog, indices, minimum):
    topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= minimum)
    queries = list(pilot.query_panel(catalog, topics))
    measurements = {pct: {arm: [] for arm in FORMATS} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        started = perf_counter()
        query = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
        hybrid = catalog.rank_for_vector(topic, query)
        rank_ms = (perf_counter() - started) * 1000
        shift = index % len(FORMATS)
        masks = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for pct in masks:
            for arm in FORMATS[shift:] + FORMATS[:shift]:
                started = perf_counter()
                title_rank = rank_title(arm, indices[arm], query)
                lookup_us = (perf_counter() - started) * 1e6
                selected = field.field_candidates(
                    hybrid, title_rank, catalog.visible[pct], catalog.person[pair],
                    full_reserved=2, use_roles=True,
                )
                values = catalog.verify(topic, selected)
                row = nomination.evaluate(catalog, topic, pair, selected, values, holdout.CUTOFF)
                row.update(
                    topic=topic, role_pair=pair, reads=len(selected), selected_sources=tuple(selected),
                    passing_sources=tuple(i for i, score in zip(selected, values, strict=True)
                                          if score > holdout.CUTOFF),
                    characters=sum(len(catalog.texts[i]) for i in selected),
                    ms=rank_ms + (perf_counter() - started) * 1000, lookup_us=lookup_us,
                )
                measurements[pct][arm].append(row)
    return measurements


def summarize(catalog, indices, minimum, prior, major):
    traces = collect(catalog, indices, minimum)
    outcomes = {}
    for pct, arms in traces.items():
        fp = arms["fp32"]
        baseline = prior[str(pct)]["role_title_four"]["compared"]["topic_four"]
        for metric in ("grounded", "cited_off_label", "issued", "reads"):
            if sum(row[metric] for row in fp) != baseline[metric]:
                raise ValueError(f"FP32 title behavior differs: {pct}, {metric}")
        if major is not None and sum(
            source in major.get(row["topic"], ()) for row in fp for _, source in row["witnesses"]
        ) != baseline["major_hits"]:
            raise ValueError("FP32 major-topic evidence differs from the published replay")
        outcomes[pct] = {}
        for arm, rows in arms.items():
            record = dict(
                compared=holdout.analyze(rows, fp),
                paired_queries=len(rows),
                same_ordered_candidates=sum(a["selected_sources"] == b["selected_sources"]
                                            for a, b in zip(rows, fp, strict=True)),
                same_ordered_recommendations=sum(a["witnesses"] == b["witnesses"]
                                                 for a, b in zip(rows, fp, strict=True)),
                title_sort_p95_us=float(np.quantile([row["lookup_us"] for row in rows], .95)),
            )
            if major is not None:
                targets = [major.get(row["topic"], set()) & catalog.person[row["role_pair"]] for row in rows]
                record["major_tagged_slots"] = sum(
                    source in target for row, target in zip(rows, targets, strict=True)
                    for _, source in row["witnesses"]
                )
                record["major_stage"] = diagnosis.stage_counts(rows, targets, catalog.bindings)
            outcomes[pct][arm] = record
    return outcomes


def run(plos, plos_manifest, elife, mesh_snapshot, reference, embedding, reranker, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    plos_meta = json.loads(plos_manifest.read_bytes())
    mesh_data = json.loads(mesh_snapshot.read_bytes())
    old = json.loads(reference.read_bytes())["run"]
    if (pilot.digest(plos), pilot.digest(elife)) != (
            plos_meta["projection_sha256"], mesh_data["projection_sha256"]):
        raise ValueError("Expected the frozen bibliographic projections")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (old["embedding"], old["reranker"]):
        raise ValueError("Expected frozen checkpoints")

    outcomes, storage = {}, {}
    for name, path in (("plos", plos), ("elife", elife)):
        catalog, title, _ = field.data_model(path, encoder, reader)
        indices = build(title)
        storage[name] = {arm: sum(array.nbytes for array in index) for arm, index in indices.items()}
        major = None
        minimum = 4
        if name == "elife":
            complete = {i for i, a in enumerate(catalog.articles) if a["declaration_complete"]}
            catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
            mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS)
            major = dict(catalog.topics)
            mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
            minimum = scope.MINIMUM_SUPPORT
            for term in tuple(catalog.topics):
                if not minimum <= len(catalog.topics[term]) <= scope.MAXIMUM_SUPPORT:
                    del catalog.topics[term]
        warm = min(catalog.topics)
        catalog.verify(warm, catalog.rank(warm)[:8])
        outcomes[name] = summarize(catalog, indices, minimum, old[f"{name}_results"], major)

    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), numpy=np.__version__,
        script_sha256=pilot.digest(Path(__file__)), mesh_snapshot_sha256=pilot.digest(mesh_snapshot),
        field_reference_sha256=pilot.digest(reference), embedding=encoder_info["revision"],
        reranker=reader.provenance["revision"], title_payload_bytes=storage, panels=outcomes,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "plos_manifest", "elife", "mesh_snapshot", "reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
