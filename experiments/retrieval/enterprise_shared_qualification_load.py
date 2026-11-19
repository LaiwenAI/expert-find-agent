"""Measure objective-first verification with paired open-loop arrival traces."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from experiments.retrieval import enterprise_role_first_query as role
from experiments.retrieval import enterprise_shared_qualification as serial
from experiments.retrieval.enterprise_cross_precision_load import workload
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("guarded", "automatic")
SCENARIOS = ((1.8, 1), (2.7, 2))
SEEDS = (20261120, 20261121)


def arrival(size, rate, passes, seed):
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(size) for _ in range(passes)]).tolist()
    offsets = np.r_[0., np.cumsum(rng.exponential(1/rate, size=len(order)-1))]
    return order, offsets


def replay(arm, fixture, texts, arrival_order, offsets):
    model_work = {step: Counter() for step in ("primary", "fallback")}
    routing = Counter()
    completed = 0

    def search(text):
        nonlocal completed
        result = serial.one(text, arm, *fixture)
        if completed:
            for step, counter in model_work.items():
                counter.update(result[step])
            routing.update({key: result["metadata"][key] for key in (
                "staged", "fallback", "original_pairs", "exact_duplicates", "removed_due_to_objective")})
        completed += 1
        detail = json.dumps(result["decisions"], sort_keys=True, separators=(",", ":")).encode()
        return dict(
            qualifications=result["qualifications"],
            retrieved=result["retrieved"],
            decisions=dict(
                initial=result["initial"], rescue=result["extra"], invoked=result["invoked"],
                ranked=result["ranked"], detail_digest=sha256(detail).hexdigest(),
            ),
        )

    offered = [texts[i] for i in arrival_order]
    count = len(offered)
    visits = list(range(count))
    split = count // 2
    long_requests = set(sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))[:len(texts)//4])
    bodies, timings = workload(
        offered, visits, offsets, search,
        strata=dict(early=set(visits[:split]), late=set(visits[split:]),
                    longest_quartile={pos for pos, i in enumerate(arrival_order) if i in long_requests}),
    )
    if completed != count + 1 or timings["completed"] != count:
        raise ValueError("Request lost or warm-up counted")
    decoded = [json.loads(body) for body in bodies]
    decisions = [r["decisions"] for r in decoded]
    detail_digests = [item.pop("detail_digest") for item in decisions]
    timings.update(
        model_work={step: dict(stats) for step, stats in model_work.items()},
        routing=dict(routing),
        issued=sum(bool(r["initial"] or r["rescue"]) for r in decisions),
        fallback_invoked=sum(r["invoked"] for r in decisions),
        added_slots=sum(len(r["rescue"]) for r in decisions),
    )
    return decoded, detail_digests, timings


def run(queries, corpus, roles, vectors, meta, models, serial_result, output):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Unexpected development-only inputs")
    prior = json.loads(serial_result.read_bytes())
    if (prior["code_sha256"], prior["role_windows_sha256"], prior["gpu"]) != (
            digest(Path(serial.__file__)), digest(roles), torch.cuda.get_device_name()):
        raise ValueError("Serial benchmark provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    requests = rows(queries)
    texts = [request["text"] for request in requests]
    if len(texts) != 80 or any(request["split"] != "development" for request in requests):
        raise ValueError("Unexpected development query inventory")
    records = rows(corpus)
    sources = {source["id"]: source for source in records}
    if len(sources) != len(records) or len(sources) != 52667:
        raise ValueError("Unexpected source inventory")
    cached = json.loads(meta.read_bytes())
    path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(path)
    vectors_hash = digest(vectors)
    if (SOURCE_INVENTORY, checkpoint["revision"], vectors_hash, len(sources)) != tuple(
            cached[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Frozen vector inventory changed")
    encoder = QwenTorchEmbedding(path)
    dense = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]
    ))
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    canonical = Reranker(models / "qwen3-reranker-0.6b")
    revisions = dict(encoder=checkpoint["revision"], verifier=canonical.provenance["revision"],
                     parser=parser.readers["bf16"].provenance["revision"])
    if revisions != prior["models"]:
        raise ValueError("Model revision changed")
    fixture = parser, encoder, dense, sources, role.load_role_index(roles, SOURCE_INVENTORY, sources), serial.EncodedReader(canonical)
    measurements = []
    for tier, (rate, passes) in enumerate(SCENARIOS):
        for repetition, seed in enumerate(SEEDS):
            order, offsets = arrival(len(texts), rate, passes, seed)
            arms = ARMS if (tier + repetition) % 2 == 0 else ARMS[::-1]
            cases, signatures, reports = {}, {}, {}
            for arm in arms:
                cases[arm], signatures[arm], reports[arm] = replay(arm, fixture, texts, order, offsets)
                print(f"Completed {arm}, nominal rate {rate}, seed {seed}, requests {len(order)}", flush=True)
            left, right = (cases[arm] for arm in ARMS)
            first, second = (signatures[arm] for arm in ARMS)
            report = dict(
                nominal_rate=rate, passes=passes, seed=seed, requests=len(order), execution_order=arms,
                issued_response_changes=sum(a != b for a, b in zip(left, right, strict=True)),
                per_criterion_changes=sum(a != b for a, b in zip(first, second, strict=True)),
                arms=reports,
            )
            measurements.append(report)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        unique_queries=len(texts), sources=len(sources), models=revisions,
        queries_sha256=DEVELOPMENT, sources_sha256=SOURCE_INVENTORY, roles_sha256=digest(roles),
        vectors_sha256=vectors_hash, serial_sha256=digest(serial_result), code_sha256=digest(Path(__file__)),
        workload_sha256=digest(Path(workload.__code__.co_filename)), traces=measurements))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial_result", "output"):
        cli.add_argument(field, type=Path)
    run(**vars(cli.parse_args()))
