"""Test local FP32 scoring of the certified title band on a fixed OpenBLAS host."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import real_title_guard as guarded
from experiments.credit import real_title_runtime as cohort
from experiments.credit import title_int8_runtime as native
from experiments.credit import topic_role_postings as pilot


BLOCK = 32


def blocked_scores(title, indices, query):
    result = np.empty(len(indices), dtype=np.float32)
    block_ids = indices // BLOCK
    for block in np.unique(block_ids):
        start = int(block) * BLOCK
        present = block_ids == block
        scores = title[start:start + BLOCK] @ query
        result[present] = scores[indices[present] - start]
    return result


def rank_band(indices, scores):
    order = np.argsort(-scores, kind="stable")
    return indices[order[:2]].tolist()


def select_local(guard, query, scorer):
    indices, proposed, ambiguous = guard.recheck_band(query)
    if not ambiguous:
        return proposed, False
    return rank_band(indices, scorer(guard.exact, indices, query)), True


def verify_scores(guard, vectors):
    report = {name: Counter() for name in ("gather", "contiguous32")}
    for query in vectors:
        indices, _, ambiguous = guard.recheck_band(query)
        reference = (guard.exact @ query)[indices].view(np.uint32)
        for name, scores in (
            ("gather", guard.exact[indices] @ query),
            ("contiguous32", blocked_scores(guard.exact, indices, query)),
        ):
            diff = scores.view(np.uint32) != reference
            report[name].update(
                checked_band_values=len(indices), different_values=int(diff.sum()),
                ambiguous_band_values=len(indices) if ambiguous else 0,
                ambiguous_different_values=int(diff.sum()) if ambiguous else 0,
            )
    return report


def run(embedding, kernel, output, prior):
    import torch

    if output.exists() or kernel.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh outputs and Slurm CUDA")
    root, papers, _, owners, source_ids, queries, _ = cohort.load_cohort()
    earlier = json.loads(prior.read_bytes())["tail_replay"]
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, max_length=128, batch_size=128)
    encoder.model.config.use_cache = False
    lib = native.compile_kernel(kernel)
    if not lib.title_vnni_available():
        raise ValueError("This replay requires AVX-512 VNNI")
    with TemporaryDirectory() as work:
        path = Path(work) / "titles.npy"
        np.save(path, encoder.encode([papers[pid]["title"] for pid in source_ids]))
        title = np.load(path, mmap_mode="r")
        guard = guarded.TitleGuard(lib, title)
        encoder.max_length = 512
        encoder.batch_size = 32

        def encode(qid):
            p = papers[qid]
            return encoder.encode([p["title"] + "\n" + p["abstract"]], query=True)[0]

        vectors = {qid: encode(qid) for qid in queries}
        scores = verify_scores(guard, vectors.values())
        def full_fallback(q):
            selected, _, rechecked = guard.select(q)
            return selected, rechecked

        routes = {
            "fp32": lambda q: (native.fp32_top(title, q, guard.eligible, 2), False),
            "full_fallback": full_fallback,
            "gather": lambda q: select_local(guard, q, lambda x, ids, v: x[ids] @ v),
            "contiguous32": lambda q: select_local(guard, q, blocked_scores),
        }
        times = {mode: {route: [] for route in routes} for mode in ("cached", "online")}
        matches = {mode: {route: Counter() for route in routes} for mode in times}
        connection = sqlite3.connect(Path(work) / "sources.sqlite")
        try:
            cohort.prepare_storage(connection, source_ids, papers, owners)
            for route in routes.values():
                cohort.render(connection, route(vectors[queries[0]])[0])
            for repeat in range(3):
                for index, qid in enumerate(queries):
                    keys = tuple(routes)
                    shift = (repeat + index) % len(keys)
                    for mode in times:
                        results = {}
                        for name in keys[shift:] + keys[:shift]:
                            started = perf_counter()
                            query = vectors[qid] if mode == "cached" else encode(qid)
                            selected, rechecked = routes[name](query)
                            response = cohort.render(connection, selected)
                            times[mode][name].append((perf_counter() - started) * 1000)
                            matches[mode][name].update(requests=1, rechecked=int(rechecked))
                            results[name] = response
                        for name, response in results.items():
                            matches[mode][name]["exact_responses"] += response == results["fp32"]
        finally:
            connection.close()
        for mode, records in matches.items():
            baseline = records["full_fallback"]
            if (baseline["requests"], baseline["exact_responses"], baseline["rechecked"]) != (
                    earlier["agreement"][mode]["requests"], earlier["agreement"][mode]["guarded_matches"],
                    earlier["exact_work"][mode]["fallback_count"]):
                raise ValueError("The guarded control did not reproduce the prior replay")
        pilot.write_json(output, dict(
            job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), numpy=np.__version__,
            title_rows=len(title), queries=len(queries), script_sha256=pilot.digest(Path(__file__)),
            source_manifest_sha256=pilot.digest(root / "manifest.json"), prior_sha256=pilot.digest(prior),
            embedding_revision=encoder_info["revision"], score_equality=scores, matches=matches,
            timings_ms={mode: {name: {
                "p50": float(np.quantile(samples, .50)), "p95": float(np.quantile(samples, .95)),
                "p99": float(np.quantile(samples, .99)),
            } for name, samples in measured.items()} for mode, measured in times.items()},
        ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "kernel", "output", "prior"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
