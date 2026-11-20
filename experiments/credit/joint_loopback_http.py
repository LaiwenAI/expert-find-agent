"""Serve the live joint evidence path through persistent loopback HTTP."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import Future, TimeoutError
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from queue import Empty, Queue
from random import Random
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_request_queue as queue
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.cof_loopback_http import Server, exchange


LIVE = queue.live
ARMS = ("serial", "batch8", "padded8")
RATES = (60, 150, 280)
CONNECTIONS = 64
REPEATS = 4
CLIENT_DEADLINE_SECONDS = 8


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path != "/search":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 16_384:
                raise ValueError("Invalid body size")
            body = json.loads(self.rfile.read(size))
            if body["panel"] not in self.server.panels:
                raise ValueError("Unknown corpus")
            requirements = body["requirements"]
            if len(requirements) != 2 or not all(
                isinstance(item[key], str) and item[key].strip()
                for item in requirements for key in ("topic", "role")
            ):
                raise ValueError("Expected two topic–role conditions")
            request = queue.Request(body["panel"], json.dumps({"requirements": requirements}))
        except (KeyError, ValueError, TypeError):
            self.send_error(400)
            return
        future = Future()
        self.server.pending.put((request, perf_counter(), future))
        try:
            response, batch, wait, compute = future.result(timeout=CLIENT_DEADLINE_SECONDS)
            payload = response.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Server-Timing", f"{batch},{wait:.6f},{compute:.6f},0")
            self.end_headers()
            self.wfile.write(payload)
        except TimeoutError:
            self.send_error(503)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.send_error(500)


def worker(pending, indexes, model, arm, warm, ready):
    capacity, hold, pad_to = queue.ARMS[arm]
    try:
        for i in range(1, capacity + 1):
            list(queue.execute(indexes, model, warm[:i], pad_to))
        ready.set_result(True)
    except Exception as exc:
        ready.set_exception(exc)
        return
    while (first := pending.get()) is not None:
        batch = [first]
        deadline = first[1] + hold
        while len(batch) < capacity:
            timeout = max(0., deadline - perf_counter())
            try:
                item = pending.get(timeout=timeout)
            except Empty:
                break
            if item is None:
                pending.put(None)
                break
            batch.append(item)
        begun = perf_counter()
        try:
            answers = queue.execute(indexes, model, [item[0] for item in batch], pad_to)
            for (_, arrived, future), (content, completed) in zip(batch, answers, strict=True):
                future.set_result((content, len(batch), (begun - arrived)*1000, (completed - begun)*1000))
        except Exception as exc:
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(exc)


def make_trace(requests, rate):
    rng = Random(20261120)
    order = []
    for _ in range(REPEATS):
        part = list(requests)
        rng.shuffle(part)
        order.extend(part)
    arrivals, at = [], 0.
    for _ in order:
        arrivals.append(at)
        at += rng.expovariate(rate)
    return order, arrivals


def wire(request):
    body = json.dumps(dict(panel=request.panel, **json.loads(request.payload))).encode()
    return (b"POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)


async def run_clients(port, trace, arrivals):
    sockets = await asyncio.gather(*(asyncio.open_connection("127.0.0.1", port) for _ in range(CONNECTIONS)))
    bodies = {request: wire(request) for request in set(trace)}
    pending = asyncio.Queue()
    records = [None] * len(trace)
    clock = asyncio.get_running_loop().time
    start = clock()

    async def client(reader, writer):
        try:
            while (position := await pending.get()) is not None:
                arrived = start + arrivals[position]
                sent = clock()
                timeout = CLIENT_DEADLINE_SECONDS - (sent - arrived)
                if timeout <= 0:
                    records[position] = {"failure": "client_deadline"}
                    continue
                try:
                    answer, batch, server_queue, compute, _ = await asyncio.wait_for(
                        exchange(reader, writer, bodies[trace[position]]), timeout=timeout,
                    )
                except (TimeoutError, ValueError, RuntimeError, ConnectionError, asyncio.IncompleteReadError) as exc:
                    records[position] = {"failure": ("client_deadline" if isinstance(exc, TimeoutError)
                                                      else type(exc).__name__)}
                    writer.close()
                    await writer.wait_closed()
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    continue
                finished = clock()
                records[position] = dict(response=answer, batch=batch, total_ms=(finished-arrived)*1000,
                                         network_ms=(finished-sent)*1000, client_queue_ms=(sent-arrived)*1000,
                                         server_queue_ms=server_queue, compute_ms=compute)
        finally:
            writer.close()
            await writer.wait_closed()

    async def produce():
        for i, at in enumerate(arrivals):
            remaining = start + at - clock()
            if remaining > 0:
                await asyncio.sleep(remaining)
            pending.put_nowait(i)
        for _ in sockets:
            pending.put_nowait(None)

    await asyncio.wait_for(asyncio.gather(produce(), *(client(*pair) for pair in sockets)), timeout=45)
    return records, clock() - start


def replay(indexes, encoder, trace, arrivals, arm):
    with Server(("127.0.0.1", 0), Handler) as server:
        server.pending = Queue()
        server.panels = indexes.keys()
        ready = Future()
        gpu = Thread(target=worker, args=(server.pending, indexes, encoder, arm, trace, ready))
        gpu.start()
        try:
            ready.result(timeout=60)
            net = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
            net.start()
            try:
                return asyncio.run(run_clients(server.server_port, trace, arrivals))
            finally:
                server.shutdown()
                net.join()
        finally:
            server.pending.put(None)
            gpu.join()


def summarize(records, elapsed, trace, arrivals, serial, padded):
    complete = [(req, row) for req, row in zip(trace, records, strict=True) if "response" in row]
    return dict(
        requests=len(trace), completed=len(complete), failures=dict(Counter(
            row["failure"] for row in records if "failure" in row
        )),
        offered_rps=(len(trace)-1)/arrivals[-1], achieved_rps=len(complete)/elapsed,
        changed_vs_serial=sum(row["response"] != serial[req] for req, row in complete),
        changed_vs_padded=sum(row["response"] != padded[req] for req, row in complete),
        batch_sizes=dict(Counter(row["batch"] for _, row in complete)),
        p95_ms={name: float(np.quantile([row[name] for _, row in complete], .95))
                for name in ("total_ms", "network_ms", "client_queue_ms", "server_queue_ms", "compute_ms")}
        if complete else {},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "output"):
        parser.add_argument(path, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on Slurm CUDA")
    checkpoint = LIVE.verify_embedding(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    plos, _ = LIVE.BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, checksum = LIVE.elife.load_catalogue(args.elife_xml, args.manifest)
    plos_subjects = LIVE.BASE.publisher_topics([args.plos_xml])
    plos_queries = LIVE.BASE.query_panel(LIVE.BASE.annotation_postings(plos, plos_subjects))
    _, _, life_queries = LIVE.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_queries), elife=(life, life_queries))
    with TemporaryDirectory() as scratch:
        indexes = {key: LIVE.LocalIndex(catalogue, encoder, Path(scratch) / key)
                   for key, (catalogue, _) in panels.items()}
        requests = [queue.Request(panel, LIVE.request(atoms)) for panel, (_, query) in panels.items() for atoms in query]
        serial = {req: next(queue.execute(indexes, encoder, [req]))[0].encode() for req in requests}
        padded = {req: next(queue.execute(indexes, encoder, [req], 8))[0].encode() for req in requests}
        results = {}
        for i, rate in enumerate(RATES):
            trace, arrivals = make_trace(requests, rate)
            offset = (args.order_offset + i) % len(ARMS)
            results[str(rate)] = {}
            for arm in ARMS[offset:] + ARMS[:offset]:
                rows, elapsed = replay(indexes, encoder, trace, arrivals, arm)
                results[str(rate)][arm] = summarize(rows, elapsed, trace, arrivals, serial, padded)
                print(f"Completed {rate} {arm}", flush=True)
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)), checkpoint=checkpoint,
        elife_projection=checksum, distinct_requests=len(requests), connections=CONNECTIONS,
        serial_vs_padded_reference=sum(serial[req] != padded[req] for req in requests),
        order_offset=args.order_offset, results=results,
    ))


if __name__ == "__main__":
    main()
