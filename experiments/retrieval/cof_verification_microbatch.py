"""Evaluate guarded and deferred verification with common cross-request GPU batching."""

import argparse
import asyncio
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import perf_counter

import torch

from expert_search.cof_benchmark import load
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import cof_live_certified_batching as paired
from experiments.retrieval import cof_loopback_http as transport
from experiments.retrieval import cof_verification_http as service
from experiments.retrieval.cof_compact_profile_shortlist import title_tokens


RATES = (4, 10)
SUPPORT = 2
REPEATS = 2
QUERIES = 48
WORKERS = 4
MAX_ROWS = 16
BUCKET = 128
WAIT_SECONDS = .001
METHODS = tuple((arm, workers) for workers in (1, WORKERS) for arm in service.ARMS)


@dataclass
class Command:
    kind: str
    work: object
    future: Future


class BatchBroker:
    """Serialize GPU work. Merge only dynamic rows with compatible lengths."""

    def __init__(self, forward):
        self.forward = forward
        self.queue = Queue()
        self.backlog = deque()
        self.work = Counter()
        self.thread = Thread(target=self.run)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.queue.put(None)
        self.thread.join()

    def submit(self, kind, work):
        answer = Future()
        self.queue.put(Command(kind, work, answer))
        return answer.result()

    def serial(self, work):
        return self.submit("serial", work)

    def fixed(self, work):
        return self.submit("fixed", work)

    def score(self, rows):
        return self.submit("score", rows)

    @staticmethod
    def bucket(rows):
        return (max(len(row["input_ids"]) for row in rows) + BUCKET - 1) // BUCKET

    def collect(self, initial):
        group = [initial]
        count = len(initial.work)
        width = self.bucket(initial.work)
        deadline = perf_counter() + WAIT_SECONDS
        waiting = self.backlog
        self.backlog = deque()
        while count < MAX_ROWS:
            try:
                other = waiting.popleft() if waiting else self.queue.get(timeout=max(0., deadline - perf_counter()))
            except Empty:
                break
            if other is None:
                self.backlog.append(other)
                break
            if (other.kind == "score" and self.bucket(other.work) == width
                    and count + len(other.work) <= MAX_ROWS):
                group.append(other)
                count += len(other.work)
            else:
                self.backlog.append(other)
            if not waiting and perf_counter() >= deadline:
                break
        self.backlog.extend(waiting)
        return group

    def run(self):
        while (job := (self.backlog.popleft() if self.backlog else self.queue.get())) is not None:
            if job.kind != "score":
                try:
                    result = job.work()
                    self.work[job.kind] += 1
                    job.future.set_result(result)
                except Exception as error:
                    job.future.set_exception(error)
                continue
            group = self.collect(job)
            try:
                values, width = self.forward([row for batch in group for row in batch.work])
                self.work["dynamic_launches"] += 1
                self.work["merged_launches"] += len(group) > 1
                self.work["dynamic_groups"] += len(group)
                self.work["dynamic_rows"] += len(values)
                self.work["dynamic_padded_tokens"] += len(values) * width
                start = 0
                for batch in group:
                    end = start + len(batch.work)
                    batch.future.set_result((values[start:end], width))
                    start = end
            except Exception as error:
                for batch in group:
                    if not batch.future.done():
                        batch.future.set_exception(error)


def request_worker(queue, measure, method, broker, jobs):
    while (item := queue.get()) is not None:
        query, arrived, future = item
        start = perf_counter()
        try:
            _, result = measure(query, SUPPORT, method, broker=broker)
            payload = service.encode_response(result["selected"])
            finish = perf_counter()
            jobs.put(dict(calls=result["calls"], padded_tokens=result["padded_tokens"],
                          scored_documents=result["scored_documents"]))
            future.set_result((payload, 1, (start-arrived)*1000, (finish-start)*1000, 0.0))
        except Exception as error:
            future.set_exception(error)


def trial(measure, method, nworkers, queries, plan):
    if nworkers == 1:
        rows, replies, runtime, jobs = service.service_trial(measure, method, queries, plan)
        return rows, replies, runtime, jobs, {}
    jobs = Queue()
    with BatchBroker(measure.forward) as broker, transport.Server(("127.0.0.1", 0), transport.Handler) as server:
        server.pending = Queue()
        measure(next(iter(queries.values())), SUPPORT, method, broker=broker)
        broker.work.clear()
        network = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        workers = [Thread(target=request_worker, args=(server.pending, measure, method, broker, jobs))
                   for _ in range(nworkers)]
        for worker in workers:
            worker.start()
        network.start()
        try:
            rows, replies, runtime = asyncio.run(transport.load(server.server_port, plan, queries))
        finally:
            server.shutdown()
            network.join()
            for _ in workers:
                server.pending.put(None)
            for worker in workers:
                worker.join()
        progress = dict(broker.work)
    completed_jobs = []
    while not jobs.empty():
        completed_jobs.append(jobs.get_nowait())
    return rows, replies, runtime, completed_jobs, progress


def selected_panel(ids, texts):
    excluded = set()
    for name in ("NIPS", "SIGIR", "SciRepEval"):
        _, papers, _, labels = load(name)
        excluded.update(tuple(title_tokens(papers[qid]["title"])) for qid in labels["hard"])
    eligible = [qid for qid in ids if tuple(title_tokens(texts[qid].split("\n", 1)[0])) not in excluded]
    order = sorted(eligible, key=lambda qid: sha256(f"kdd-microbatch-v1:{qid}".encode()).digest())
    if len(order) < QUERIES:
        raise ValueError("Expected at least 48 title-disjoint KDD queries")
    return {qid: texts[qid] for qid in order[:QUERIES]}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.dataset = "KDD"
    args.exclude_nips_titles = False
    measure, ids, texts, _, inputs, encoder, reranker = paired.prepare_runner(args)
    queries = selected_panel(ids, texts)
    expected = {}
    for qid, query in queries.items():
        _, result = measure(query, SUPPORT, "exhaustive")
        expected[qid] = service.encode_response(result["selected"])
    print("Canonical KDD panel complete", flush=True)
    raw = {}
    for index, rate in enumerate(RATES):
        runs = {}
        for repeat in range(REPEATS):
            plan = service.plan_for(queries, rate, repeat)
            rotation = (index+repeat) % len(METHODS)
            for method, nworkers in METHODS[rotation:] + METHODS[:rotation]:
                label = f"{method}_w{nworkers}"
                rows, replies, runtime, jobs, broker = trial(measure, method, nworkers, queries, plan)
                if not len(rows) == len(replies) == len(jobs) == len(queries):
                    raise AssertionError("Missing HTTP responses or request records")
                exact = sum(reply[0] == expected[row["query"]]
                            for row, reply in zip(rows, replies, strict=True))
                runs.setdefault(label, []).append(dict(records=rows, jobs=jobs, runtime=runtime,
                                                       exact_responses=exact, broker=broker))
                print(rate, repeat, label, "exact", exact, "merged", broker.get("merged_launches", 0), flush=True)
        raw[rate] = runs
    report = {rate: {arm: service.summarize(runs) for arm, runs in arms.items()} for rate, arms in raw.items()}
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), inputs=inputs,
        script_sha256=digest(Path(__file__)), runner_sha256=digest(Path(paired.__file__)),
        service_sha256=digest(Path(service.__file__)), transport_sha256=digest(Path(transport.__file__)),
        checkpoint=encoder, reranker=reranker, query_ids=list(queries),
        plan=dict(rates=RATES, repeats=REPEATS, support=SUPPORT, workers=(1, WORKERS),
                  max_rows=MAX_ROWS, bucket_tokens=BUCKET, gather_ms=1000*WAIT_SECONDS,
                  no_cross_request_score_cache=True),
        report=report, raw=raw,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(argument, type=Path)
    run(parser.parse_args())
