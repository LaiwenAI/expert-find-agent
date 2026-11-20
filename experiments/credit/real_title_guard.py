"""Certify ordered title citations with an INT8 interval scan and exact boundary rechecks."""

import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import real_title_runtime as previous
from experiments.credit import title_index_compression as compression
from experiments.credit import title_int8_runtime as native
from experiments.credit import topic_role_postings as pilot


def gamma(dim, dtype):
    # Use a conservative sum of dim products and up to dim additions.
    error = 2 * dim * np.finfo(dtype).eps / 2
    return error / (1 - error)


class TitleGuard:
    def __init__(self, lib, exact):
        self.exact = exact
        self.codes, scales = compression.quantize(exact)
        self.scales = np.ascontiguousarray(scales[:, 0])
        self.sums = self.codes.sum(axis=1, dtype=np.int32)
        self.residuals = np.empty(len(exact), dtype=np.float32)
        self.max_source_norm = 0.
        for start in range(0, len(exact), 1024):
            part = slice(start, start + 1024)
            originals = exact[part].astype(np.float64)
            reconstructed = self.codes[part] * self.scales[part, None].astype(np.float64)
            residual = np.linalg.norm(originals - reconstructed, axis=1)
            self.residuals[part] = np.nextafter(residual.astype(np.float32), np.float32(np.inf))
            self.max_source_norm = max(self.max_source_norm, float(np.max(np.linalg.norm(originals, axis=1))))
        self.max_source_norm += 1e-12
        self.max_quant_norm = self.max_source_norm + float(self.residuals.max())
        self.eligible = np.ones(len(exact), dtype=np.uint8)
        self.upper = np.empty(len(exact), dtype=np.float64)
        self.ids = np.empty(len(exact), dtype=np.int32)
        self.call = lib.title_top_two_guard
        ptr = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
        self.call.argtypes = [ptr(d) for d in (np.int8, np.float32, np.int32, np.float32, np.int8, np.uint8)]
        self.call.argtypes += [ctypes.c_float, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                               ptr(np.float64), ptr(np.int32)]
        self.call.restype = ctypes.c_int

    @property
    def payload_bytes(self):
        return sum(arr.nbytes for arr in (self.codes, self.scales, self.sums, self.residuals))

    def select(self, query):
        codes, scale = compression.quantize(query)
        shifted = (codes.astype(np.int16) + 128).astype(np.uint8)
        fp64_query = query.astype(np.float64)
        query_norm = float(np.linalg.norm(fp64_query)) + 1e-12
        query_error = float(np.linalg.norm(fp64_query - codes.astype(np.float64) * float(scale[0]))) + 1e-12
        fp32_error = gamma(len(query), np.float32) * self.max_source_norm * query_norm
        numeric_slack = 1e-11 * (1 + self.max_source_norm * query_norm)
        band = self.call(self.codes, self.scales, self.sums, self.residuals, codes, shifted, scale[0],
                         query_norm, self.max_quant_norm * query_error + fp32_error + numeric_slack,
                         *self.codes.shape, self.upper, self.ids)
        if not 2 <= band <= len(self.exact):
            raise ValueError(f"Invalid precision band size: {band}")
        indices = self.ids[:band]
        scores = self.exact[indices].astype(np.float64) @ fp64_query
        best = np.argsort(-scores, kind="stable")[:3]
        # Check both response order and membership against any FP32 evaluation of these rows.
        within_band_error = fp32_error + gamma(len(query), np.float64) * self.max_source_norm * query_norm
        fallback = bool(np.min(scores[best[:-1]] - scores[best[1:]]) <= 2 * within_band_error + numeric_slack)
        selected = native.fp32_top(self.exact, query, self.eligible, 2) if fallback else indices[best[:2]].tolist()
        return selected, band, fallback


def run(embedding, kernel, output, prior_results):
    import torch

    if output.exists() or kernel.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh outputs and Slurm CUDA")
    root, papers, _, owners, source_ids, queries, _ = previous.load_cohort()
    expected = json.loads(prior_results.read_bytes())["determinism_check"]
    provenance = pilot.verify_checkpoint(embedding)
    if provenance["revision"] != expected["embedding_revision"] or pilot.digest(root / "manifest.json") != expected[
        "source_manifest_sha256"
    ]:
        raise ValueError("Expected the original SciRepEval title replay inputs")
    encoder = QwenTorchEmbedding(embedding, max_length=128, batch_size=128)
    encoder.model.config.use_cache = False
    lib = native.compile_kernel(kernel)
    if not lib.title_vnni_available():
        raise ValueError("This replay requires AVX-512 VNNI")
    with TemporaryDirectory() as work:
        title_path = Path(work) / "title.npy"
        np.save(title_path, encoder.encode([papers[pid]["title"] for pid in source_ids]))
        title = np.load(title_path, mmap_mode="r")
        guard = TitleGuard(lib, title)
        encoder.max_length = 512
        encoder.batch_size = 32

        def encode(qid):
            paper = papers[qid]
            return encoder.encode([paper["title"] + "\n" + paper["abstract"]], query=True)[0]

        vectors = {qid: encode(qid) for qid in queries}
        methods = {
            "fp32": lambda v: (native.fp32_top(title, v, guard.eligible, 2), 0, False),
            "int8": lambda v: (native.native_top(lib, guard.codes, guard.scales, guard.sums, v, guard.eligible, 2, True),
                               0, False),
            "guarded_int8": guard.select,
        }
        samples = {mode: {name: [] for name in methods} for mode in ("cached", "online")}
        matched = {mode: Counter() for mode in samples}
        work_counts = {mode: dict(bands=[], fallbacks=[]) for mode in samples}
        stable = {}
        connection = sqlite3.connect(Path(work) / "sources.sqlite")
        try:
            previous.prepare_storage(connection, source_ids, papers, owners)
            for method in methods.values():
                previous.render(connection, method(vectors[queries[0]])[0])
            for repeat in range(3):
                for index, qid in enumerate(queries):
                    keys = tuple(methods)
                    shift = (repeat + index) % len(keys)
                    for mode, measured in samples.items():
                        by_method = {}
                        for name in keys[shift:] + keys[:shift]:
                            start = perf_counter()
                            vector = vectors[qid] if mode == "cached" else encode(qid)
                            selected, band, fallback = methods[name](vector)
                            response = previous.render(connection, selected)
                            elapsed = (perf_counter() - start) * 1000
                            measured[name].append(elapsed)
                            by_method[name] = response
                            if stable.setdefault((qid, name), response) != response:
                                raise ValueError("Response changed across encoding modes or replay repeats")
                            if name == "guarded_int8":
                                work_counts[mode]["bands"].append(band)
                                if fallback:
                                    work_counts[mode]["fallbacks"].append(elapsed)
                        if by_method["guarded_int8"] != by_method["fp32"]:
                            raise ValueError(f"The certified path changed the response for {qid}")
                        matched[mode].update(requests=1, guarded_matches=1,
                                             raw_matches=by_method["int8"] == by_method["fp32"])
        finally:
            connection.close()

        for mode, counts in matched.items():
            if (counts["requests"], counts["raw_matches"]) != (
                    expected["agreement"][mode]["requests"], expected["agreement"][mode]["equal_response"]):
                raise ValueError("The unguarded comparison differs from the original replay")
        pilot.write_json(output, dict(
            job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), titles=len(title), queries=len(queries),
            script_sha256=pilot.digest(Path(__file__)),
            kernel_sha256=pilot.digest(Path(native.__file__).with_name("title_int8_kernel.cpp")),
            prior_results_sha256=pilot.digest(prior_results), agreement=matched,
            int8_payload_bytes=guard.payload_bytes, fp32_backing_bytes=title.nbytes,
            timings_ms={mode: {k: dict(p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)),
                                                       p99=float(np.quantile(v, .99)), maximum=max(v))
                               for k, v in methods.items()} for mode, methods in samples.items()},
            exact_work={mode: dict(
                band_p50=float(np.quantile(work["bands"], .5)),
                band_p95=float(np.quantile(work["bands"], .95)), band_max=max(work["bands"]),
                fallback_count=len(work["fallbacks"]),
                fallback_p50_ms=float(np.median(work["fallbacks"])) if work["fallbacks"] else None,
            ) for mode, work in work_counts.items()},
        ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "kernel", "output", "prior_results"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
