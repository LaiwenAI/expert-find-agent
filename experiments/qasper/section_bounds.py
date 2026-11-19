"""Audit exact and heuristic section pruning on human-annotated QASPER evidence."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter, perf_counter_ns

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.qasper.adaptive_evidence import ASSETS, digest, evidence_sets, offline, quality, query_input, read_data


TOP = 5
BEAM = 2
REPEATS = 5
METHODS = ("flat", "centroid_beam", "centroid_min5", "bounded_sections")


@dataclass
class Sections:
    vectors: np.ndarray
    leaves: tuple
    centers: np.ndarray
    radii: np.ndarray

    @classmethod
    def build(cls, vectors, sections):
        groups = defaultdict(list)
        for i, section in enumerate(sections):
            groups[section].append(i)
        leaves = tuple((np.asarray(indices), np.ascontiguousarray(vectors[indices]))
                       for indices in groups.values())
        centers = np.stack([leaf.mean(axis=0) for _, leaf in leaves])
        radii = np.asarray([np.linalg.norm(leaf - center, axis=1).max()
                            for (_, leaf), center in zip(leaves, centers, strict=True)])
        return cls(vectors, leaves, centers, radii)

    def rank(self, query, method, top=TOP):
        if method == "flat":
            scores = self.vectors @ query
            order = np.argsort(-scores, kind="stable")[:top]
            return order.tolist(), len(scores), len(self.leaves)

        centers = self.centers @ query
        if method in ("centroid_beam", "centroid_min5"):
            frontier = np.argsort(-centers, kind="stable")
            if method == "centroid_beam":
                frontier = frontier[:BEAM]
            bounds = None
        else:
            # Cauchy-Schwarz on each section's largest member-to-center distance.
            # Float64 plus conservative slack keeps rounded bounds outward.
            bounds = centers + np.linalg.norm(query) * self.radii + 1e-10
            frontier = np.argsort(-bounds, kind="stable")
        winners = []
        checks = opened = 0
        for section in frontier:
            if bounds is not None and len(winners) == top and bounds[section] < -winners[-1][0]:
                break
            indices, leaf = self.leaves[section]
            scores = leaf @ query
            winners.extend((-float(score), int(i)) for score, i in zip(scores, indices, strict=True))
            winners = sorted(winners)[:top]
            checks += len(indices)
            opened += 1
            if method == "centroid_min5" and opened >= BEAM and len(winners) == top:
                break
        return [i for _, i in winners], checks, opened


def compare(index, query, paper_id, qa, paragraphs, offset):
    gold = evidence_sets(qa, paragraphs)
    row = dict(paper_id=paper_id, question_id=qa["question_id"], total=len(paragraphs),
               sections=len(index.leaves), arms={})
    timings = defaultdict(list)
    for repeat in range(REPEATS):
        rotation = (offset + repeat) % len(METHODS)
        for method in METHODS[rotation:] + METHODS[:rotation]:
            start = perf_counter_ns()
            order, checks, leaves = index.rank(query, method)
            timings[method].append((perf_counter_ns() - start) / 1000.0)
            row["arms"][method] = dict(order=order, checks=checks, leaves=leaves)
    if row["arms"]["flat"]["order"] != row["arms"]["bounded_sections"]["order"]:
        raise AssertionError(f"Bounded section search changed dense Top-{TOP}")
    for method, result in row["arms"].items():
        selected = [paragraphs[i] for i in result["order"]]
        result["median_us"] = float(np.median(timings[method]))
        result["human_evidence"] = quality(selected, gold)
        result["exact_dense"] = result["order"] == row["arms"]["flat"]["order"]
    return row


def summarize(rows):
    answerable = [r for r in rows if r["arms"]["flat"]["human_evidence"] is not None]
    return dict(
        questions=len(rows), evidence_evaluable=len(answerable),
        papers=len({r["paper_id"] for r in rows}), mean_paragraphs=mean(r["total"] for r in rows),
        methods={
            arm: dict(
                exact_dense_top5=sum(r["arms"][arm]["exact_dense"] for r in rows),
                mean_checked_paragraphs=mean(r["arms"][arm]["checks"] for r in rows),
                mean_opened_sections=mean(r["arms"][arm]["leaves"] for r in rows),
                any_paragraph_skipped=sum(r["arms"][arm]["checks"] < r["total"] for r in rows),
                fewer_than_top5=sum(len(r["arms"][arm]["order"]) < TOP for r in rows),
                cpu_p95_us=float(np.quantile([r["arms"][arm]["median_us"] for r in rows], .95)),
                complete_evidence=mean(r["arms"][arm]["human_evidence"]["complete_evidence"] for r in answerable),
                evidence_f1=mean(r["arms"][arm]["human_evidence"]["f1"] for r in answerable),
                complete_delta_vs_flat=clustered_paired_interval([
                    r["arms"][arm]["human_evidence"]["complete_evidence"]
                    - r["arms"]["flat"]["human_evidence"]["complete_evidence"] for r in answerable
                ], [r["paper_id"] for r in answerable]),
            ) for arm in METHODS
        },
    )


def run(data, model, output):
    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm GPU allocation is required")
    test = read_data(data, "test")
    model_info = json.loads((model / "checkpoint-provenance.json").read_bytes())
    if model_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected checkpoint")
    for name, checksum in model_info["files"].items():
        if digest(model / name) != checksum:
            raise ValueError(f"Changed checkpoint: {name}")
    encoder = QwenTorchEmbedding(model, max_length=512, batch_size=32)
    if not encoder.torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    encoder.model.config.use_cache = False
    encoder.encode([query_input("What method is used?")])
    metadata, _, vectors, source_timing = offline(test, encoder)
    print("Source vectors complete", source_timing, flush=True)
    questions = [(pid, qa) for pid, paper in test.items() for qa in paper["qas"]]
    start = perf_counter()
    query_vectors = encoder.encode([query_input(qa["question"]) for _, qa in questions])
    query_encoding_seconds = perf_counter() - start
    start = perf_counter()
    bank = {
        pid: Sections.build(np.ascontiguousarray(vectors[lo:hi], dtype=np.float64), sections)
        for pid, (_, sections, lo, hi) in metadata.items()
    }
    compilation_seconds = perf_counter() - start
    rows = [compare(bank[pid], np.asarray(vector, dtype=np.float64), pid, qa, metadata[pid][0], i)
            for i, ((pid, qa), vector) in enumerate(zip(questions, query_vectors, strict=True))]
    report = summarize(rows)
    output.mkdir()
    raw_path = output / "rows.json"
    raw_path.write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(dict(
        model=model_info, gpu=encoder.torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
        script_sha256=digest(Path(__file__)), data_sha256=ASSETS["qasper-test-v0.3.json"],
        source_timing=source_timing, query_encoding_seconds=query_encoding_seconds,
        index_compilation_seconds=compilation_seconds, timing_repeats=REPEATS,
        rows_sha256=digest(raw_path), report=report,
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
