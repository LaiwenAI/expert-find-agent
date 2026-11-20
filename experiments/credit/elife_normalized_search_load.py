"""Replay bursty arrivals against two equivalent query-normalization policies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Queue
from time import perf_counter, sleep

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import elife_role_normalization_search as search
from experiments.credit import study_role_normalization as roles
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


ARMS = ("normalize_per_source", "normalize_per_query")
RATES = (2., 7., 11.)


def schedules(rows, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows)).tolist()
    arrival = np.r_[0., np.cumsum(rng.exponential(size=len(rows) - 1))]
    return order, arrival / (arrival[-1] / (len(rows) - 1))


def run_trace(catalog, judge, rows, order, arrivals, arm):
    pending = Queue()
    response = []
    ready = Queue()

    def serve():
        ready.put(True)
        for _ in order:
            index, scheduled, enqueued = pending.get()
            started = perf_counter()
            result = search.run_query(catalog, judge, rows[index], arm)
            wire = json.dumps(result["delivered"], ensure_ascii=False, separators=(",", ":")).encode()
            completed = perf_counter()
            response.append(dict(
                index=index, scheduled=scheduled, enqueued=enqueued, started=started,
                completed=completed, payload=wire, role_calls=result["role_model_calls"],
                issued=result["issued"], tagged=result["tag_backed"], wrong_role=result["wrong_role"],
            ))

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(serve)
        ready.get()
        origin = perf_counter() + .02
        for index, shift in zip(order, arrivals, strict=True):
            due = origin + shift
            delay = due - perf_counter()
            if delay > 0:
                sleep(delay)
            pending.put((index, due, perf_counter()))
        worker.result()
    if [row["index"] for row in response] != order:
        raise ValueError("Unexpected FIFO response order")
    return response


def summarize(response):
    def p95(end, start):
        return float(np.quantile([(row[end] - row[start]) * 1000 for row in response], .95))

    finished = np.asarray([row["completed"] for row in response])
    positions = [(len(response) * pct // 4) - 1 for pct in (1, 2, 3, 4)]
    unfinished = [int(i + 1 - np.count_nonzero(finished <= response[i]["scheduled"])) for i in positions]
    wire = sha256()
    for row in response:
        wire.update(len(row["payload"]).to_bytes(4, "big"))
        wire.update(row["payload"])
    return dict(
        requests=len(response), serial_service_p95_ms=p95("completed", "started"),
        schedule_to_completion_p95_ms=p95("completed", "scheduled"),
        queue_p95_ms=p95("started", "enqueued"),
        ingress_lateness_p95_ms=p95("enqueued", "scheduled"),
        unfinished_at_arrival_quartiles=unfinished,
        drain_ms=(response[-1]["completed"] - response[-1]["scheduled"]) * 1000,
        payload_sha256=wire.hexdigest(),
        **{k: sum(row[k] for row in response) for k in ("role_calls", "issued", "tagged", "wrong_role")},
    )


def main():
    import torch

    from experiments.followir.core17_verifier import Reranker

    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "reference", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--replay", type=int, choices=(0, 1), default=0)
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    spec = json.loads(args.manifest.read_bytes())
    prior = json.loads(args.reference.read_bytes())["run"]
    if digest(args.projection) != spec["projection_sha256"]:
        raise ValueError("Unexpected eLife projection")
    embedding = Path("artifacts/models/qwen3-embedding-0.6b")
    reranker = Path("artifacts/models/qwen3-reranker-0.6b")
    info = pilot.verify_checkpoint(embedding)
    enc = QwenTorchEmbedding(embedding)
    enc.model.config.use_cache = False
    reader = Reranker(reranker)
    if (info["revision"], reader.provenance["revision"]) != (prior["embedding"], prior["reranker"]):
        raise ValueError("Unexpected topical checkpoints")
    judge = Judge()
    roles.extended_ids(judge)
    articles = [dict(row, topics=row["keywords"]) for row in json.loads(args.projection.read_bytes())["sources"]
                if row["declaration_complete"]]
    catalog = pilot.Catalog(articles, enc, reader)
    rows = list(search.query_set(catalog))
    order, unit = schedules(rows, 20261120 + args.replay)
    for arm in ARMS:
        search.run_query(catalog, judge, rows[0], arm)
    results = {}
    for i, rate in enumerate(RATES):
        arms = ARMS if (args.replay + i) % 2 == 0 else tuple(reversed(ARMS))
        responses = {arm: run_trace(catalog, judge, rows, order, unit / rate, arm) for arm in arms}
        left, right = (responses[arm] for arm in ARMS)
        if [row["payload"] for row in left] != [row["payload"] for row in right]:
            raise ValueError("Normalization reuse changed a response")
        results[rate] = {arm: summarize(responses[arm]) for arm in ARMS}
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), replay=args.replay,
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection),
        reference_sha256=digest(args.reference), decision_model_sha256=judge.sha,
        arrival_profiles=results,
    ))


if __name__ == "__main__":
    main()
