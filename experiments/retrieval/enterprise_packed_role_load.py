"""Replay warm enterprise packed verification under paired loopback arrivals."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval import enterprise_role_first_query as role_search
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("padded_whole", "padded_rescue", "guarded_rescue")
SEEDS = (20261119, 20261120)
RATE = 3.0


def replay(arm, fixture, texts, offered_order, offsets):
    # Keep one worker per device. This counter belongs to that worker.
    completed = 0
    work = {stage: Counter() for stage in ("primary", "fallback")}

    def search(text):
        nonlocal completed

        result = packed.one(text, arm, *fixture)
        if completed:
            for name, counter in work.items():
                counter.update(result[name])
        completed += 1
        # The loopback wrapper passes these three fields over the wire.
        # Include both the primary and rescue citations in exact comparisons.
        return dict(
            qualifications=result["qualifications"], retrieved=result["ranked"],
            decisions=dict(initial=result["initial"], primary_detail=result["decisions"], fallback=result["extra"],
                           invoked=result["invoked"], clipped=result["clipped"]),
        )

    from experiments.retrieval.enterprise_cross_precision_load import workload

    longest = set(sorted(range(len(texts)), key=lambda idx: (-len(texts[idx]), idx))[:len(texts) // 4])
    bodies, timings = workload(
        texts, offered_order, offsets, search,
        strata=dict(longest_quartile=longest, remainder=set(range(len(texts))) - longest),
    )
    if completed != len(texts) + 1 or timings["completed"] != len(texts):
        raise ValueError("A playback request was lost or a warm-up call was counted")
    decoded = [json.loads(body) for body in bodies]
    decision = [row["decisions"] for row in decoded]
    timings.update(
        model_work={name: dict(counter) for name, counter in work.items()},
        issued=sum(bool(row["initial"] or row["fallback"]) for row in decision),
        fallback_invoked=sum(row["invoked"] for row in decision),
        added_slots=sum(len(row["fallback"]) for row in decision),
        clipped=sum(row["clipped"] for row in decision),
    )
    return bodies, timings


def run(queries, corpus, roles, vectors, meta, models, serial, output):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_cross_precision_load import trace
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    role_search.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Use the pinned development inventory")
    frozen = json.loads(serial.read_bytes())
    if frozen["code_sha256"] != digest(Path(packed.__file__)) or frozen["role_windows_sha256"] != digest(roles):
        raise ValueError("The serial experiment uses a different implementation or role inventory")
    torch.backends.cuda.matmul.allow_tf32 = False
    requests = rows(queries)
    texts = [row["text"] for row in requests]
    sources = review.index(rows(corpus), lambda item: item["id"])
    if (len(texts), len(sources)) != (80, 52667):
        raise ValueError("Unexpected cohort size")
    saved = json.loads(meta.read_bytes())
    path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(path)
    verified = digest(vectors)
    if (saved["corpus_sha256"], saved["checkpoint"], saved["rows"], saved["vectors_sha256"]) != (
        SOURCE_INVENTORY, checkpoint["revision"], len(sources), verified
    ) or verified != frozen["vectors_sha256"]:
        raise ValueError("Unexpected source cache")
    embedding = QwenTorchEmbedding(path)
    reference = ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[src["expert_id"]] for src in sources.values()],
    )
    index = GpuOwnedVectorIndex(reference)
    role_index = role_search.load_role_index(roles, SOURCE_INVENTORY, sources)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    canonical = Reranker(models / "qwen3-reranker-0.6b")
    revisions = dict(encoder=checkpoint["revision"], verifier=canonical.provenance["revision"],
                     parser=parser.readers["bf16"].provenance["revision"])
    if revisions != frozen["models"] or torch.cuda.get_device_name() != frozen["gpu"]:
        raise ValueError("Unexpected accelerator or model revision")
    reader = packed.Reader(canonical)
    fixture = (parser, embedding, index, sources, role_index, reader)

    traces = []
    for repetition, seed in enumerate(SEEDS):
        order, times = trace(len(texts), RATE, seed)
        arms = ARMS if repetition == 0 else ARMS[::-1]
        replies, report = {}, {}
        for arm in arms:
            replies[arm], report[arm] = replay(arm, fixture, texts, order, times)
            print(f"Completed {arm} on arrival seed {seed}", flush=True)
        changed = sum(a != b for a, b in zip(replies["guarded_rescue"], replies["padded_rescue"], strict=True))
        traces.append(dict(seed=seed, execution_order=arms, responses_changed_guarded_vs_padded=changed, arms=report))
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), requests=len(texts), sources=len(sources),
        role_windows=len(role_index[0]), nominal_rate=RATE, models=revisions, queries_sha256=DEVELOPMENT,
        source_sha256=SOURCE_INVENTORY, vectors_sha256=verified, serial_sha256=digest(serial),
        code_sha256=digest(Path(__file__)), traces=traces,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
