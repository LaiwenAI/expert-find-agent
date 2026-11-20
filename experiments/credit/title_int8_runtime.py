"""Compare fixed-budget title retrieval with a native INT8 kernel and fused top-k."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import role_field_ranking as field
from experiments.credit import title_index_compression as compression
from experiments.credit import topic_role_postings as pilot


def compile_kernel(build_path):
    source = Path(__file__).with_name("title_int8_kernel.cpp")
    subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off",
                    str(source), "-o", str(build_path)], check=True)
    lib = ctypes.CDLL(str(build_path.resolve()))
    lib.title_vnni_available.restype = ctypes.c_bool
    function = lib.title_top_int8
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
        for dtype in (np.int8, np.float32, np.int32, np.int8, np.uint8)
    ] + [ctypes.c_float, np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
         np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
    function.restype = ctypes.c_int
    return lib


def native_top(lib, codes, scales, sums, vector, eligible, limit, use_vnni):
    query, factor = compression.quantize(vector)
    shifted = (query.astype(np.int16) + 128).astype(np.uint8)
    output = np.empty(limit, dtype=np.int32)
    count = lib.title_top_int8(codes, scales, sums, query, shifted, factor[0], eligible,
                              *codes.shape, limit, use_vnni, output)
    if not 0 <= count <= limit:
        raise ValueError("Native kernel returned an invalid output count")
    return output[:count].tolist()


def anchors_and_mask(hybrid, opaque, posting):
    flags = opaque.copy()
    flags[posting] = 1
    anchor = []
    for source in hybrid:
        if flags[source]:
            anchor.append(source)
            if len(anchor) == 2:
                break
    flags[anchor] = 0
    return anchor, flags


def fill_budget(picked, hybrid):
    for source in hybrid:
        if len(picked) == 4:
            break
        if source not in picked:
            picked.append(source)
    return picked


def fp32_top(title, query, flags, limit):
    scores = title @ query
    scores[flags == 0] = -np.inf
    selected = []
    for _ in range(limit):
        source = int(np.argmax(scores))
        if not np.isfinite(scores[source]):
            break
        selected.append(source)
        scores[source] = -np.inf
    return selected


def latency(samples):
    return {name: dict(
        mean_us=float(np.mean(values)), p95_us=float(np.quantile(values, .95)),
    ) for name, values in samples.items()}


def real_catalog(catalog, title, queries, lib, use_vnni):
    codes, scales = compression.build(title)["int8"]
    sums = codes.sum(axis=1, dtype=np.int32)
    samples = {name: [] for name in ("fp32_sort", "fp32_bounded", "numpy_int8_sort", "native_scalar", "native_vnni")}
    counts = {pct: dict(requests=0, equal_fp32=0, equal_numpy=0) for pct in pilot.CACHE}
    opaque = {
        pct: np.array([i not in known for i in range(len(title))], dtype=np.uint8)
        for pct, known in catalog.visible.items()
    }
    for index, (topic, _, pair) in enumerate(queries):
        query = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
        hybrid = catalog.rank_for_vector(topic, query)
        posting = catalog.person[pair]
        keys = list(samples)
        shift = index % len(keys)
        for pct in pilot.CACHE:
            chosen = {}
            for name in keys[shift:] + keys[:shift]:
                started = perf_counter()
                if name.endswith("_sort"):
                    arm = "fp32" if name.startswith("fp32") else "int8"
                    ranked = compression.rank_title(arm, (title,) if arm == "fp32" else (codes, scales), query)
                    picked = field.field_candidates(hybrid, ranked, catalog.visible[pct], posting,
                                                    full_reserved=2, use_roles=True)
                else:
                    if name == "native_vnni" and not use_vnni:
                        continue
                    anchor, flags = anchors_and_mask(hybrid, opaque[pct], list(posting))
                    limit = 4 - len(anchor)
                    if name == "fp32_bounded":
                        added = fp32_top(title, query, flags, limit)
                    else:
                        added = native_top(lib, codes, scales, sums, query, flags, limit, name == "native_vnni")
                    picked = fill_budget(anchor + added, hybrid)
                samples[name].append((perf_counter() - started) * 1e6)
                chosen[name] = picked
            if chosen["fp32_bounded"] != chosen["fp32_sort"]:
                raise ValueError("The bounded FP32 path differs from the stable sort")
            baseline = chosen["numpy_int8_sort"]
            if any(picked != baseline for name, picked in chosen.items() if name.startswith("native")):
                raise ValueError("Native route changed the original INT8 candidate order")
            counts[pct]["requests"] += 1
            counts[pct]["equal_numpy"] += 1
            counts[pct]["equal_fp32"] += baseline == chosen["fp32_sort"]
    return dict(masks=counts, selection=latency({k: v for k, v in samples.items() if v}),
                payload_bytes=int(codes.nbytes + scales.nbytes + sums.nbytes))


def stress(title, query_vectors, lib, use_vnni):
    results = {}
    for copies in (1, 16, 128):
        expanded = np.tile(title, (copies, 1))
        codes, scales = compression.quantize(expanded)
        scales = np.ascontiguousarray(scales[:, 0])
        sums = codes.sum(axis=1, dtype=np.int32)
        eligible = np.ones(len(codes), dtype=np.uint8)
        samples = {name: [] for name in ("fp32_sort", "fp32_bounded", "numpy_int8_sort", "native_scalar", "native_vnni")}
        for index, query in enumerate(query_vectors):
            keys = list(samples)
            shift = index % len(keys)
            chosen = {}
            for name in keys[shift:] + keys[:shift]:
                started = perf_counter()
                if name == "fp32_sort":
                    picked = compression.rank_title("fp32", (expanded,), query)[:2]
                elif name == "fp32_bounded":
                    picked = fp32_top(expanded, query, eligible, 2)
                elif name == "numpy_int8_sort":
                    picked = compression.rank_title("int8", (codes, scales), query)[:2]
                else:
                    if name == "native_vnni" and not use_vnni:
                        continue
                    picked = native_top(lib, codes, scales, sums, query, eligible, 2, name == "native_vnni")
                samples[name].append((perf_counter() - started) * 1e6)
                chosen[name] = picked
            if chosen["fp32_bounded"] != chosen["fp32_sort"]:
                raise ValueError("The bounded FP32 path differs from the stable sort")
            if any(picked != chosen["numpy_int8_sort"]
                   for name, picked in chosen.items() if name.startswith("native")):
                raise ValueError("Native stage changed exact top-2 INT8 order")
        results[len(expanded)] = dict(
            requests=len(query_vectors), selection=latency({k: v for k, v in samples.items() if v}),
            fp32_bytes=expanded.nbytes, native_bytes=codes.nbytes + scales.nbytes + sums.nbytes,
        )
    return results


def run(projection, snapshot, compression_reference, embedding, library, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    locked = json.loads(snapshot.read_bytes())
    prior = json.loads(compression_reference.read_bytes())["run"]
    if pilot.digest(projection) != locked["projection_sha256"] or pilot.digest(snapshot) != prior["mesh_snapshot_sha256"]:
        raise ValueError("The title catalog differs from the pinned compression study")
    encoder_info = pilot.verify_checkpoint(embedding)
    if encoder_info["revision"] != prior["embedding"]:
        raise ValueError("Expected the previous encoder")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    catalog, title, _ = field.data_model(projection, encoder, None)
    complete = {i for i, a in enumerate(catalog.articles) if a["declaration_complete"]}
    catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
    mesh.assign_topics(catalog, locked["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
    topics = sorted(topic for topic, ids in catalog.topics.items() if
                    scope.MINIMUM_SUPPORT <= len(ids) <= scope.MAXIMUM_SUPPORT)
    queries = list(pilot.query_panel(catalog, topics))
    lib = compile_kernel(library)
    use_vnni = lib.title_vnni_available()
    # Warm all scoring paths before recording the fixed request sequence.
    real_catalog(catalog, title, queries[:2], lib, use_vnni)
    real = real_catalog(catalog, title, queries, lib, use_vnni)
    for pct, actual in real["masks"].items():
        expected = prior["panels"]["elife"][str(pct)]["int8"]
        if (actual["requests"], actual["equal_fp32"]) != (
                expected["paired_queries"], expected["same_ordered_candidates"]):
            raise ValueError("The title comparison no longer reproduces the frozen candidate counts")
    stress_vectors = [encoder.encode([f"Research about {topic}"], query=True)[0] for topic in topics[:16]] * 3
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), architecture=platform.machine(),
        vnni=use_vnni, numpy=np.__version__, script_sha256=pilot.digest(Path(__file__)),
        kernel_sha256=pilot.digest(Path(__file__).with_name("title_int8_kernel.cpp")),
        snapshot_sha256=pilot.digest(snapshot), compression_reference_sha256=pilot.digest(compression_reference),
        real_title_candidates=real, synthetic_title_lookup=stress(title, stress_vectors, lib, use_vnni),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "snapshot", "compression_reference", "embedding", "library", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
