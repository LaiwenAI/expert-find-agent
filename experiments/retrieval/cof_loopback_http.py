"""Replay CoF retrieval load through persistent loopback HTTP connections."""

import argparse
import asyncio
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from queue import Empty, Queue
import socket
import sqlite3
from threading import Thread
import time

import numpy as np
import torch

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_request_batching import load_plan, report
from experiments.retrieval.cof_resumable_query import payload

CONNECTIONS = 64
RATES = (120, 240)
METHODS = {'serial': 1, 'microbatch8': 8}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, _format, *_args):
        pass

    def do_POST(self):
        if self.path != '/search':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 32768:
                raise ValueError('Invalid request length')
            query = json.loads(self.rfile.read(length))['query']
            if not isinstance(query, str) or not query.strip():
                raise ValueError('Missing query')
        except (TypeError, KeyError, ValueError):
            self.send_error(400)
            return
        future = Future()
        submitted = time.perf_counter()
        self.server.pending.put((query, submitted, future))
        response, batch, queued, serving, encoded = future.result(timeout=20)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('X-Server-Timing', f'{batch},{queued:.6f},{serving:.6f},{encoded:.6f}')
        self.end_headers()
        self.wfile.write(response)


class Server(ThreadingHTTPServer):
    request_queue_size = 128

    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address


def worker(pending, ranker, database, encoder, cap, warm_query, ready):
    try:
        with sqlite3.connect(database) as connection:
            encoder.batch_size = cap
            for size in (1, cap):
                vectors = encoder.encode([warm_query] * size, query=True)
                for vector in vectors:
                    serialize_hits(ranker, connection, ranker.rank_host(ranker.score(vector)))
            ready.set_result(True)
            while (first := pending.get()) is not None:
                batch = [first]
                for _ in range(cap - 1):
                    try:
                        batch.append(pending.get_nowait())
                    except Empty:
                        break
                begun = time.perf_counter()
                try:
                    vectors = encoder.encode([item[0] for item in batch], query=True)
                    encoded = time.perf_counter()
                    for (_, arrived, future), vector in zip(batch, vectors, strict=True):
                        rank = ranker.rank_host(ranker.score(vector))
                        response, _ = serialize_hits(ranker, connection, rank)
                        finished = time.perf_counter()
                        future.set_result((response, len(batch), (begun - arrived) * 1000,
                                           (finished - begun) * 1000, (encoded - begun) * 1000))
                except Exception as error:
                    for _, _, future in batch:
                        if not future.done():
                            future.set_exception(error)
    except Exception as error:
        if not ready.done():
            ready.set_exception(error)
        else:
            raise


async def exchange(reader, writer, request):
    writer.write(request)
    await writer.drain()
    status = await reader.readline()
    if not status.startswith(b'HTTP/1.1 200 '):
        raise RuntimeError(f'Unexpected HTTP status {status!r}')
    headers = {}
    while (line := await reader.readline()) != b'\r\n':
        key, value = line.split(b':', 1)
        headers[key.lower()] = value.strip()
    response = await reader.readexactly(int(headers[b'content-length']))
    timing = headers[b'x-server-timing'].split(b',')
    return response, int(timing[0]), *(float(value) for value in timing[1:])


async def load(port, plan, queries):
    arrivals, ids = plan
    requests = {}
    for qid, text in queries.items():
        body = payload(text).encode('utf-8')
        requests[qid] = (b'POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                         + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
    sockets = await asyncio.gather(*(asyncio.open_connection('127.0.0.1', port) for _ in range(CONNECTIONS)))
    pending = asyncio.Queue()
    records, responses = [None] * len(ids), [None] * len(ids)
    loop = asyncio.get_running_loop()
    start = loop.time()

    async def client(reader, writer):
        try:
            while (item := await pending.get()) is not None:
                index, released = item
                qid = ids[index]
                sent = loop.time()
                answer, batch, queued, service, encoded = await exchange(reader, writer, requests[qid])
                finished = loop.time()
                arrived = start + arrivals[index]
                records[index] = {'query': qid, 'batch': batch, 'queue_ms': queued,
                                  'service_ms': service, 'total_ms': (finished - arrived) * 1000,
                                  'batch_encode_ms': encoded,
                                  'arrival_timer_ms': (released - arrived) * 1000,
                                  'client_queue_ms': (sent - released) * 1000,
                                  'http_roundtrip_ms': (finished - sent) * 1000}
                responses[index] = (answer, [hit['expert'] for hit in json.loads(answer)])
        finally:
            writer.close()
            await writer.wait_closed()

    async def produce():
        for i, at in enumerate(arrivals):
            delay = start + at - loop.time()
            while delay > 0:
                await asyncio.sleep(delay)
                delay = start + at - loop.time()
            pending.put_nowait((i, loop.time()))
        for _ in sockets:
            pending.put_nowait(None)

    await asyncio.wait_for(asyncio.gather(produce(), *(client(*s) for s in sockets)), timeout=30)
    elapsed = loop.time() - start
    return records, responses, {'elapsed_seconds': elapsed,
                                'drained_requests_per_second': len(ids) / elapsed,
                                'issued_requests_per_second': (len(ids) - 1) / arrivals[-1]}


def compare_method(ranker, database, encoder, mode, plan, queries):
    with Server(('127.0.0.1', 0), Handler) as server:
        server.pending = Queue()
        ready = Future()
        gpu = Thread(target=worker, args=(server.pending, ranker, database, encoder, METHODS[mode],
                                         next(iter(queries.values())), ready))
        gpu.start()
        try:
            ready.result(timeout=60)
            net = Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
            net.start()
            try:
                return asyncio.run(load(server.server_port, plan, queries))
            finally:
                server.shutdown()
                net.join()
        finally:
            server.pending.put(None)
            gpu.join()


def stats(records, responses, baseline, labels, runtime):
    data = report(records, responses, baseline, labels, runtime)
    for field in ('arrival_timer_ms', 'client_queue_ms', 'http_roundtrip_ms'):
        data['latency_ms'][field] = {'p95': float(np.quantile([r[field] for r in records], .95))}
    return data


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside a Slurm GPU allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, digest in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != digest:
            raise ValueError('Changed encoder')
    variant = QueryEncoderVariants()
    variant.configure('no_cache')
    encoder = variant.encoder
    omitted_queries, omitted_sources, _ = exclusions()
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for i, dataset in enumerate(('SIGIR', DATASET)):
        options = {} if not i else {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources}
        ranker, db, queries, labels, info = prepare(dataset, output, checkpoint, **options)
        by_rate = {}
        try:
            for j, rate in enumerate(RATES):
                plan = load_plan(queries, rate)
                methods = list(METHODS) if (i + j) % 2 == 0 else list(reversed(METHODS))
                arms = {method: compare_method(ranker, db, encoder, method, plan, queries) for method in methods}
                reference = arms['serial'][1]
                by_rate[rate] = {}
                for method, (records, responses, runtime) in arms.items():
                    metrics = stats(records, responses, reference, labels, runtime)
                    path = output / f'{dataset}-{rate}-{method}.json'
                    path.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
                    metrics['rows_sha256'] = sha(path)
                    by_rate[rate][method] = metrics
                print(dataset, rate, 'complete', flush=True)
        finally:
            db.unlink()
        datasets[dataset] = {'index': info, 'loads': by_rate}
        del ranker
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'checkpoint_sha256': sha(checkpoint), 'script_sha256': sha(Path(__file__)),
              'datasets': datasets, 'connections': CONNECTIONS, 'methods': METHODS,
              'scope': 'Warm persistent loopback HTTP/1.1 with one GPU worker on a single node. '
                       'Client and server share a process. Fixed reproducible arrival traces and '
                       'request bodies use actual TCP sockets. Includes HTTP, scheduling, query '
                       'encoding, exact retrieval, citation lookup and serialization. '
                       'Excludes external network, model startup and semantic evidence verification.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
