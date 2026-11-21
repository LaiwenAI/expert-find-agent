"""Measure warmed enterprise first-query serving under open-loop arrivals."""

import argparse
import asyncio
from concurrent.futures import Future
import json
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_loopback_http import Handler, Server, exchange
from experiments.retrieval.enterprise_cross_precision_search import Parser
from experiments.retrieval.enterprise_first_query import evaluate, read_rows
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY


RATES = (1.5, 3.)
SEEDS = (20261119, 20261120)
ARMS = ("bf16_guard", "consensus")
WARMUP = "Find an expert in optical inspection. Prior experience testing polymer films is required."


class RequestHandler(Handler):
    def do_POST(self):
        if self.path != "/search":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 32768:
                raise ValueError("Invalid size")
            text = json.loads(self.rfile.read(length))["query"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid request")
        except (KeyError, TypeError, ValueError):
            self.send_error(400)
            return
        future = Future()
        self.server.pending.put((text, perf_counter(), future))
        try:
            body, queued, service = future.result(timeout=180)
        except Exception:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Server-Timing", f"1,{queued:.6f},{service:.6f},0")
        self.end_headers()
        self.wfile.write(body)


def wire_result(result):
    return json.dumps({name: result[name] for name in ("qualifications", "retrieved", "decisions")},
                      ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def serve(pending, search, ready):
    try:
        search(WARMUP)
    except Exception as exc:
        ready.set_exception(exc)
        return
    ready.set_result(None)
    while (item := pending.get()) is not None:
        text, arrived, future = item
        begun = perf_counter()
        try:
            body = wire_result(search(text))
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result((body, (begun-arrived)*1000, (perf_counter()-begun)*1000))


def trace(size, rate, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size).tolist()
    gaps = rng.exponential(1/rate, size=size-1)
    offsets = np.r_[0., np.cumsum(gaps)]
    return indices, offsets


async def send_trace(port, texts, indices, times):
    loop = asyncio.get_running_loop()
    start = loop.time()
    async def issue(index, offset):
        await asyncio.sleep(max(0., start+offset-loop.time()))
        released = loop.time()
        body = json.dumps({"query":texts[index]}, ensure_ascii=False).encode()
        request = (b"POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                   + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            result, _, queued, service, _ = await asyncio.wait_for(exchange(reader, writer, request), timeout=180)
            done = loop.time()-start
            return result, dict(total_ms=(done-offset)*1000, queue_ms=queued, service_ms=service,
                                release_lag_ms=(released-start-offset)*1000, done=done)
        finally:
            writer.close()
            await writer.wait_closed()
    return await asyncio.gather(*(issue(i,t) for i,t in zip(indices,times,strict=True)))


def workload(texts, indices, times, search, *, pending_factory=Queue, strata=None, timeline=False, worker_fn=serve):
    pending, ready = pending_factory(), Future()
    worker = Thread(target=worker_fn, args=(pending,search,ready))
    with Server(("127.0.0.1",0), RequestHandler) as server:
        server.pending = pending
        worker.start()
        try:
            ready.result(timeout=90)
            listener = Thread(target=server.serve_forever)
            listener.start()
            try:
                result = asyncio.run(send_trace(server.server_port, texts, indices, times))
            finally:
                server.shutdown()
                listener.join()
        finally:
            pending.put(None)
            worker.join()
    bodies, rows = zip(*result, strict=True)
    end = max(row["done"] for row in rows)
    report = dict(
        completed=len(rows), final_completion_seconds=end,
        drain_seconds=max(0., end-float(times[-1])),
        observed_offered_rate=(len(rows)-1)/times[-1],
        completed_per_second=len(rows)/end,
        quantiles_ms={field:np.quantile([row[field] for row in rows],[.5,.95,.99]).tolist()
                      for field in ("total_ms","queue_ms","service_ms","release_lag_ms")},
    )
    if strata is not None:
        report["strata"] = {
            name: {
                "requests": len(subset),
                "total_p50_p95_max_ms": np.quantile([r["total_ms"] for r in subset], [.5, .95, 1.]).tolist(),
                "queue_p95_max_ms": np.quantile([r["queue_ms"] for r in subset], [.95, 1.]).tolist(),
            }
            for name, subset in ((name, [row for index, row in zip(indices, rows, strict=True) if index in members])
                                 for name, members in strata.items())
            if subset
        }
    if timeline:
        report["timings"] = rows
    return bodies, report



def run(queries, corpus, models, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU node")
    if (digest(queries),digest(corpus)) != (DEVELOPMENT,SOURCE_INVENTORY):
        raise ValueError("Changed development data")
    requests, records = read_rows(queries), read_rows(corpus)
    texts = [r["text"] for r in requests]
    sources = {r["id"]:r for r in records}
    if len(texts)!=80 or len(sources)!=len(records) or len(records)!=52667:
        raise ValueError("Unexpected cohort")
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = models/"qwen3-embedding-0.6b"
    embedding_meta = verify_checkpoint(model_path)
    embedding = QwenTorchEmbedding(model_path)
    start = perf_counter()
    data = embedding.encode([r["text"] for r in records])
    reference = ExactOwnedVectorIndex(data, list(sources), [[r["expert_id"]] for r in records])
    index = GpuOwnedVectorIndex(reference)
    index_seconds = perf_counter()-start
    print(f"Indexed {len(sources)} sources",flush=True)
    parser = Parser(models/"qwen3-4b-instruct-2507")
    verifier = Reranker(models/"qwen3-reranker-0.6b")
    traces = []
    for rate in RATES:
        for repeat, seed in enumerate(SEEDS):
            indices, times = trace(len(texts),rate,seed)
            ordered = ARMS if repeat==0 else ARMS[::-1]
            responses, report = {}, {}
            for arm in ordered:
                responses[arm], report[arm] = workload(
                    texts,indices,times,lambda text: evaluate(text,arm,parser,embedding,index,sources,verifier),
                )
                print(rate, seed, arm, report[arm]["completed"], flush=True)
            changed = sum(a!=b for a,b in zip(responses[ARMS[0]],responses[ARMS[1]],strict=True))
            traces.append(dict(rate=rate, seed=seed, responses_changed=changed, arms=report))
    write_json(output,dict(
        development_sha256=DEVELOPMENT,corpus_sha256=SOURCE_INVENTORY,script_sha256=digest(Path(__file__)),
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(), requests=len(texts),sources=len(sources),
        experts=len(reference.owners),index_seconds=index_seconds,
        embedding_revision=embedding_meta["revision"],verifier_revision=verifier.provenance["revision"],
        parser_revision=parser.readers["bf16"].provenance["revision"],traces=traces,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries","corpus","models","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
