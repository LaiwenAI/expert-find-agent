"""Compare matched-budget title, title-abstract, and dual-view citation retrieval."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import real_title_block_recheck as block
from experiments.credit import real_title_guard as guarded
from experiments.credit import real_title_runtime as cohort
from experiments.credit import title_int8_runtime as native
from experiments.credit import topic_role_postings as pilot


def quality_results(selections, owners, source_ids, labels):
    estimates = {}
    rng = np.random.default_rng(20261120)
    paired_draws = rng.integers(0, len(selections), size=(10000, len(selections)))
    for rubric, by_query in labels.items():
        totals = {arm: Counter() for arm in next(iter(selections.values()))}
        positives = {arm: [] for arm in totals}
        for qid, arms in selections.items():
            for arm, rows in arms.items():
                grade = {rubric: Counter()}
                cohort.quality(rows, owners, source_ids, {rubric: by_query}, qid, grade)
                totals[arm].update(grade[rubric])
                positives[arm].append(int(grade[rubric]["has_known_positive"]))
        difference = np.array(positives["dual_fp32"]) - positives["title_abstract"]
        estimates[rubric] = dict(
            totals=totals,
            dual_minus_title_abstract=dict(
                gained_queries=int((difference > 0).sum()),
                lost_queries=int((difference < 0).sum()),
                proportion_delta=float(difference.mean()),
                percentile_interval=np.quantile(difference[paired_draws].mean(axis=1), [.025, .975]).tolist(),
            ),
        )
    return estimates


def run(embedding, scratch, output, prior):
    import torch

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    root, papers, labels, owners, source_ids, queries, _ = cohort.load_cohort()
    previous = json.loads(prior.read_bytes())
    provenance = pilot.verify_checkpoint(embedding)
    if (pilot.digest(root / "manifest.json"), provenance["revision"]) != (
            previous["source_manifest_sha256"], previous["embedding_revision"]):
        raise ValueError("Expected the frozen corpus and checkpoint")

    encoder = QwenTorchEmbedding(embedding, max_length=128, batch_size=128)
    encoder.model.config.use_cache = False

    def text(pid):
        return papers[pid]["title"] + "\n" + papers[pid]["abstract"]

    with TemporaryDirectory(dir=scratch) as work:
        work = Path(work)
        kernel = native.compile_kernel(work / "title.so")
        if not kernel.title_vnni_available():
            raise ValueError("Expected AVX-512 VNNI")
        t = perf_counter()
        path = work / "titles.npy"
        np.save(path, encoder.encode([papers[pid]["title"] for pid in source_ids]))
        titles = np.load(path, mmap_mode="r")
        title_encoding_s = perf_counter() - t
        guard = guarded.TitleGuard(kernel, titles)

        encoder.max_length = 512
        encoder.batch_size = 32
        t = perf_counter()
        full = encoder.encode([text(pid) for pid in source_ids])
        full_encoding_s = perf_counter() - t

        def encode(qid):
            return encoder.encode([text(qid)], query=True)[0]

        def full_top(vector):
            return native.fp32_top(full, vector, guard.eligible, 2)

        def title_top(vector):
            return native.fp32_top(titles, vector, guard.eligible, 2)

        def dual(vector, score_title):
            first = full_top(vector)[0]
            second = next(row for row in score_title(vector) if row != first)
            return [first, second]

        def guarded_top(vector):
            return block.select_local(guard, vector, block.blocked_scores)[0]

        routes = {
            "title": title_top,
            "title_abstract": full_top,
            "dual_fp32": lambda vector: dual(vector, title_top),
            "dual_guarded": lambda vector: dual(vector, guarded_top),
        }
        vectors = {qid: encode(qid) for qid in queries}
        score_checks = block.verify_scores(guard, vectors.values())["contiguous32"]
        if score_checks["different_values"]:
            raise ValueError("Contiguous scoring differs from the reference backend")
        samples = {mode: {arm: [] for arm in routes} for mode in ("cached", "online")}
        matches = Counter()
        selected_by_query = {}
        connection = sqlite3.connect(work / "sources.sqlite")
        try:
            cohort.prepare_storage(connection, source_ids, papers, owners)
            for route in routes.values():
                cohort.render(connection, route(vectors[queries[0]]))
            order = tuple(routes)
            for repeat in range(3):
                for index, qid in enumerate(queries):
                    shift = (index + repeat) % len(order)
                    for mode, measurements in samples.items():
                        selected, responses = {}, {}
                        for arm in order[shift:] + order[:shift]:
                            t = perf_counter()
                            vector = vectors[qid] if mode == "cached" else encode(qid)
                            selected[arm] = routes[arm](vector)
                            responses[arm] = cohort.render(connection, selected[arm])
                            measurements[arm].append((perf_counter() - t) * 1000)
                        if selected_by_query.setdefault(qid, selected) != selected:
                            raise ValueError("Source selection changed across repetitions or modes")
                        if responses["dual_guarded"] != responses["dual_fp32"]:
                            raise ValueError("Guarded dual view changed the serialized citations")
                        matches[mode] += 1
        finally:
            connection.close()
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), numpy=np.__version__,
        embedding_revision=provenance["revision"], source_manifest_sha256=pilot.digest(root / "manifest.json"),
        script_sha256=pilot.digest(Path(__file__)), prior_sha256=pilot.digest(prior),
        title_rows=len(titles), queries=len(queries), citations_per_query=2, exact_dual_responses=matches,
        title_encoding_seconds=title_encoding_s, title_abstract_encoding_seconds=full_encoding_s,
        title_fp32_bytes=titles.nbytes, title_abstract_fp32_bytes=full.nbytes,
        title_guard_payload_bytes=guard.payload_bytes,
        title_score_checks=score_checks,
        dual_differs_from_title_abstract=sum(
            set(row["dual_fp32"]) != set(row["title_abstract"]) for row in selected_by_query.values()
        ),
        reviewer_proxy=quality_results(selected_by_query, owners, source_ids, labels),
        timings_ms={mode: {arm: {
            "p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)),
        } for arm, values in measurements.items()} for mode, measurements in samples.items()},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "scratch", "output", "prior"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
