"""Compare evidence-verification schedules under a shared loopback HTTP workload."""

import argparse
import asyncio
from concurrent.futures import Future
import json
from pathlib import Path
from queue import Queue
from random import Random
from statistics import mean
from threading import Thread
from time import perf_counter
import os

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import cof_live_certified_batching as paired
from experiments.retrieval import cof_loopback_http as transport


RATES = (4, 8)
ARMS = ("guarded_cached", "guarded_bucketed", "deferred_cached")
REPEATS = 2
SUPPORT = 2
LATENCY_TARGET_MS = 500
SEED = 20261119


def encode_response(selected):
    return json.dumps([dict(expert=expert, evidence=docs) for expert, docs in selected],
                      ensure_ascii=True, separators=(",", ":")).encode()


def gpu_worker(queue, measure, method, warmup, ready, jobs):
    try:
        measure(warmup, SUPPORT, method)
    except Exception as error:
        ready.set_exception(error)
        return
    ready.set_result(True)
    while (item := queue.get()) is not None:
        query, arrived, future = item
        begun = perf_counter()
        try:
            _, result = measure(query, SUPPORT, method)
            payload = encode_response(result["selected"])
            finished = perf_counter()
            jobs.append(dict(calls=result["calls"], score_calls=result["score_calls"],
                             rechecks=result["recheck_calls"], scored_documents=result["scored_documents"],
                             padded_tokens=result["padded_tokens"]))
            future.set_result((payload, 1, (begun - arrived) * 1000, (finished - begun) * 1000, 0.0))
        except Exception as error:
            future.set_exception(error)


def service_trial(measure, method, queries, plan):
    jobs = []
    with transport.Server(("127.0.0.1", 0), transport.Handler) as server:
        server.pending = Queue()
        ready = Future()
        worker = Thread(target=gpu_worker, args=(server.pending, measure, method,
                                                next(iter(queries.values())), ready, jobs))
        worker.start()
        try:
            ready.result(timeout=60)
            network = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
            network.start()
            try:
                records, answers, runtime = asyncio.run(transport.load(server.server_port, plan, queries))
            finally:
                server.shutdown()
                network.join()
        finally:
            server.pending.put(None)
            worker.join()
    return records, answers, runtime, jobs


def plan_for(queries, rate, repeat):
    order = list(queries)
    Random(SEED + rate * REPEATS + repeat).shuffle(order)
    return [i / rate for i in range(len(order))], order


def summarize(repeats):
    latency = [row for run in repeats for row in run["records"]]
    statistics = {
        field: dict(p50=float(np.quantile([r[field] for r in latency], .50)),
                    p95=float(np.quantile([r[field] for r in latency], .95)))
        for field in ("total_ms", "service_ms", "queue_ms", "arrival_timer_ms", "client_queue_ms")
    }
    jobs = [job for run in repeats for job in run["jobs"]]
    elapsed = sum(run["runtime"]["elapsed_seconds"] for run in repeats)
    deadline_hits = sum(row["total_ms"] <= LATENCY_TARGET_MS for row in latency)
    return dict(requests=len(latency), exact_responses=sum(run["exact_responses"] for run in repeats),
                latency_ms=statistics, drained_rps=len(latency)/elapsed,
                completed_within_500_ms=deadline_hits, deadline_goodput_rps=deadline_hits/elapsed,
                mean_gpu_work={key: mean(job[key] for job in jobs) for key in jobs[0]})


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    args.dataset = "SciRepEval"
    args.exclude_nips_titles = True
    measure, ids, texts, labels, inputs, encoder, reranker = paired.prepare_runner(args)
    if len(ids) != 59:
        raise ValueError("Changed title-disjoint SciRepEval cohort")
    queries = {qid: texts[qid] for qid in ids}
    expected = {}
    # Establish a fresh canonical reference in this allocation.
    for qid, query in queries.items():
        _, outcome = measure(query, SUPPORT, "exhaustive")
        expected[qid] = encode_response(outcome["selected"])
    print("Canonical reference complete", flush=True)
    raw = {}
    for j, rate in enumerate(RATES):
        trials = {arm: [] for arm in ARMS}
        for repeat in range(REPEATS):
            plan = plan_for(queries, rate, repeat)
            rotation = (j + repeat) % len(ARMS)
            for method in ARMS[rotation:] + ARMS[:rotation]:
                rows, answers, runtime, jobs = service_trial(measure, method, queries, plan)
                if len(rows) != len(jobs) or len(answers) != len(ids):
                    raise AssertionError("Requests disappeared under load")
                equality = sum(answer[0] == expected[row["query"]]
                               for row, answer in zip(rows, answers, strict=True))
                trials[method].append(dict(records=rows, jobs=jobs, runtime=runtime, exact_responses=equality))
                print(rate, repeat, method, "exact", equality, flush=True)
        raw[rate] = trials
    report = {rate: {method: summarize(trials) for method, trials in table.items()} for rate, table in raw.items()}
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), inputs=inputs,
        script_sha256=digest(Path(__file__)), verifier_script_sha256=digest(Path(paired.__file__)),
        transport_sha256=digest(Path(transport.__file__)), checkpoint=encoder, reranker=reranker,
        design=dict(rates=RATES, repeats=REPEATS, support=SUPPORT, latency_target_ms=LATENCY_TARGET_MS,
                    panel_size=len(ids), seed=SEED, connections=transport.CONNECTIONS,
                    workers=1, query_result_cache=False, cross_request_microbatch=False),
        report=report, raw=raw,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(argument, type=Path)
    run(parser.parse_args())
