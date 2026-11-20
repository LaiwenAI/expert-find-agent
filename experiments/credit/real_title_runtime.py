"""Compare native INT8 title retrieval with bounded FP32 on the CoF SciRepEval index."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.cof_benchmark import load
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import title_index_compression as compression
from experiments.credit import title_int8_runtime as native
from experiments.credit import topic_role_postings as pilot


def title_key(title):
    return " ".join(title.casefold().split())


def load_cohort():
    root, papers, profiles, labels = load("SciRepEval")
    manifest = json.loads((root / "manifest.json").read_bytes())
    owners = {}
    for reviewer, profile in profiles.items():
        for paper in set(profile["papers"]):
            owners.setdefault(paper, []).append(reviewer)
    source_ids = sorted(paper for paper in owners if papers[paper]["title"].strip())
    existing_titles = {title_key(papers[paper]["title"]) for paper in source_ids}
    excluded = set(manifest["exact_query_texts_in_profiles"]) | set(manifest["query_ids_in_profiles"])
    queries = [qid for qid in sorted(labels["soft"])
               if qid not in excluded and qid not in owners and title_key(papers[qid]["title"]) not in existing_titles]
    if len(source_ids) != 73257 or len(queries) != 91:
        raise ValueError("SciRepEval catalog or leakage-filtered query set changed")
    return root, papers, labels, owners, source_ids, queries, existing_titles


def prepare_storage(connection, source_ids, papers, owners):
    connection.execute("CREATE TABLE evidence(i INTEGER PRIMARY KEY, payload TEXT)")
    connection.executemany("INSERT INTO evidence VALUES (?,?)", (
        (index, json.dumps(dict(id=pid, title=papers[pid]["title"], excerpt=papers[pid]["abstract"][:512],
                               reviewers=sorted(owners[pid])), ensure_ascii=False, separators=(",", ":")))
        for index, pid in enumerate(source_ids)
    ))
    connection.commit()


def render(connection, selected):
    citations = [json.loads(connection.execute(
        "SELECT payload FROM evidence WHERE i=?", (row,)).fetchone()[0]) for row in selected]
    return json.dumps(citations, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def quality(selected, owners, source_ids, labels, query, totals):
    experts = {owner for row in selected for owner in owners[source_ids[row]]}
    for rubric, by_query in labels.items():
        gold = by_query[query]
        totals[rubric].update(
            has_known_positive=any(gold.get(e, 0) > 0 for e in experts),
            known_positive=sum(gold.get(e, 0) > 0 for e in experts),
            judged=sum(e in gold for e in experts), total=len(experts),
        )


def run(embedding, kernel, output):
    import torch

    if output.exists() or kernel.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh outputs and Slurm CUDA")
    root, papers, labels, owners, source_ids, queries, existing_titles = load_cohort()
    provenance = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, max_length=128, batch_size=128)
    encoder.model.config.use_cache = False
    started = perf_counter()
    title_matrix = encoder.encode([papers[pid]["title"] for pid in source_ids])
    title_encoding_seconds = perf_counter() - started
    encoder.max_length = 512
    encoder.batch_size = 32
    codes, scales = compression.quantize(title_matrix)
    scales = np.ascontiguousarray(scales[:, 0])
    sums = codes.sum(axis=1, dtype=np.int32)
    eligible = np.ones(len(title_matrix), dtype=np.uint8)
    lib = native.compile_kernel(kernel)
    if not lib.title_vnni_available():
        raise ValueError("The compared native route requires AVX-512 VNNI")
    methods = {
        "fp32_bounded": lambda query: native.fp32_top(title_matrix, query, eligible, 2),
        "int8_vnni": lambda query: native.native_top(lib, codes, scales, sums, query, eligible, 2, True),
    }

    def encode(qid):
        p = papers[qid]
        return encoder.encode([p["title"] + "\n" + p["abstract"]], query=True)[0]

    queries_encoded = {qid: encode(qid) for qid in queries}
    canonical_responses = {}
    samples = {mode: {key: [] for key in methods} for mode in ("cached", "online")}
    agreement = {mode: Counter() for mode in samples}
    grades = {key: {rubric: Counter() for rubric in labels} for key in methods}
    with TemporaryDirectory() as work:
        connection = sqlite3.connect(Path(work) / "sources.sqlite")
        try:
            prepare_storage(connection, source_ids, papers, owners)
            for lookup in methods.values():
                render(connection, lookup(queries_encoded[queries[0]]))
            for repeat in range(3):
                for index, qid in enumerate(queries):
                    order = tuple(methods)
                    if (repeat + index) % 2:
                        order = tuple(reversed(order))
                    for mode, measurements in samples.items():
                        rows = {}
                        for name in order:
                            start = perf_counter()
                            vector = queries_encoded[qid] if mode == "cached" else encode(qid)
                            selected = methods[name](vector)
                            result = render(connection, selected)
                            measurements[name].append((perf_counter() - start) * 1000)
                            expected = canonical_responses.setdefault((qid, name), result)
                            if expected != result:
                                raise ValueError("The request response changed across repeats or encoding modes")
                            rows[name] = selected, result
                            if repeat == 0 and mode == "online":
                                quality(selected, owners, source_ids, labels, qid, grades[name])
                        first, second = (rows[name] for name in methods)
                        agreement[mode].update(
                            requests=1, equal_first=int(first[0][0] == second[0][0]),
                            equal_source_set=int(set(first[0]) == set(second[0])),
                            equal_response=int(first[1] == second[1]),
                        )
        finally:
            connection.close()
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)),
        kernel_source_sha256=pilot.digest(Path(native.__file__).with_name("title_int8_kernel.cpp")),
        source_manifest_sha256=pilot.digest(root / "manifest.json"),
        embedding_revision=provenance["revision"],
        titles=len(source_ids), normalized_distinct_titles=len(existing_titles), queries=len(queries),
        title_encoding_seconds=title_encoding_seconds,
        fp32_bytes=title_matrix.nbytes, int8_bytes=codes.nbytes + scales.nbytes + sums.nbytes,
        agreement=agreement,
        unique_queries_with_changed_response=sum(
            canonical_responses[qid, "fp32_bounded"] != canonical_responses[qid, "int8_vnni"] for qid in queries
        ),
        reviewer_proxy=grades,
        timings_ms={mode: {method: dict(p50=float(np.quantile(rows, .5)), p95=float(np.quantile(rows, .95)))
                           for method, rows in measurements.items()}
                    for mode, measurements in samples.items()},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "kernel", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
