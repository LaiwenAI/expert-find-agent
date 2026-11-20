"""Measure remote declaration access for the exact structured expert path."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
from itertools import combinations
import json
import os
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter, sleep

import numpy as np

from experiments.credit import role_column_index as columns
from experiments.credit.budgeted_declaration_index import FOLDS, cold_workload, state
from experiments.credit.declaration_index_publisher_transfer import inventory
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.cof_loopback_http import Server


ARMS = ("whole_author", "whole_cold", "column_smoothed")
PROFILES = ((1, 0), (4, 0), (4, 2))


def receive(conn, method, path, body=None):
    conn.request(method, path, body, {"Content-Type": "application/json"} if body else {})
    response = conn.getresponse()
    payload = response.read()
    if response.status != 200:
        raise ValueError(f"Unexpected loopback status {response.status}")
    return payload, response.headers


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def respond(self, payload, headers=None):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(payload)


class SourceHandler(Handler):
    def do_GET(self):
        file = self.server.files.get(self.path)
        if file is None:
            self.send_error(404)
            return
        if self.server.delay:
            sleep(self.server.delay)
        self.respond(file.read_bytes())


class SearchHandler(Handler):
    def setup(self):
        super().setup()
        self.source = HTTPConnection("127.0.0.1", self.server.source_port, timeout=10)

    def finish(self):
        self.source.close()
        super().finish()

    def do_POST(self):
        plan = self.server.plans.get(self.path)
        if plan is None:
            self.send_error(404)
            return
        try:
            ask = 0
            size = int(self.headers["Content-Length"])
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            role_names = body["roles"]
            if len(role_names) != 3 or len(set(role_names)) != 3:
                raise ValueError("Expected three distinct roles")
            for name in role_names:
                ask |= self.server.roles[name]
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return

        bytes_read = 0

        def load(doc):
            nonlocal bytes_read
            payload, _ = receive(self.source, "GET", self.server.source_urls[doc])
            bytes_read += len(payload)
            return json.loads(payload)

        matches, reads = columns.partial_rank(self.server.owners, self.server.caps, plan, ask, load)
        payload = json.dumps({"matches": matches}, separators=(",", ":")).encode()
        self.respond(payload, {"X-Source-Reads": reads, "X-Source-Bytes": bytes_read})


@contextmanager
def serve(handler, **fields):
    with Server(("127.0.0.1", 0), handler) as service:
        for name, value in fields.items():
            setattr(service, name, value)
        thread = Thread(target=service.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        try:
            yield service.server_port
        finally:
            service.shutdown()
            thread.join()


def bodies(roles, held, reference, fold):
    return [
        (fold, json.dumps({"roles": [name for name, bit in roles.items() if ask & bit]}).encode(),
         json.dumps({"matches": reference[ask]}, separators=(",", ":")).encode())
        for ask in held
    ]


def execute(port, requests, arm, clients):
    def replay(batch):
        scores = []
        with closing(HTTPConnection("127.0.0.1", port, timeout=10)) as conn:
            for item in batch[:4]:
                fold, body, expected = item
                content, _ = receive(conn, "POST", f"/{fold}/{arm}", body)
                if content != expected:
                    raise AssertionError("Warm service returned different exact citations")
            for fold, body, expected in batch:
                start = perf_counter()
                content, headers = receive(conn, "POST", f"/{fold}/{arm}", body)
                elapsed = (perf_counter() - start) * 1000
                if content != expected:
                    raise AssertionError("HTTP changed the expert ranking or exact citations")
                scores.append((elapsed, int(headers["X-Source-Reads"]), int(headers["X-Source-Bytes"])))
        return scores

    with ThreadPoolExecutor(max_workers=clients) as pool:
        rows = [item for stream in pool.map(replay, [requests[i::clients] for i in range(clients)])
                for item in stream]
    return dict(
        requests=len(rows), exact_responses=len(rows),
        source_reads=sum(r[1] for r in rows), source_payload_bytes=sum(r[2] for r in rows),
        response_p50_ms=float(np.quantile([r[0] for r in rows], .50)),
        response_p95_ms=float(np.quantile([r[0] for r in rows], .95)),
    )


def run(corpus, manifest, prior, scratch, output, publisher, order_offset):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a fresh output on a Slurm compute node")
    anchor = json.loads(prior.read_bytes())["publishers"][publisher]
    expected = (columns.bundles.FIXTURE_SHA256 if publisher == "plos"
                else json.loads(manifest.read_bytes())["projection_sha256"])
    if digest(corpus) != expected or anchor["fixture_sha256"] != expected:
        raise ValueError("Unexpected publisher projection")
    if digest(Path(columns.__file__)) != anchor["script_sha256"]:
        raise ValueError("Index planner changed since the warm replay")
    roles = {name: 1 << i for i, name in enumerate(sorted(ROLES))}
    asks = [sum(group) for group in combinations(roles.values(), 3)]
    results = {}
    with TemporaryDirectory(dir=scratch, prefix="role-column-http-") as directory:
        owners, labels, caps, files, _, _ = inventory(
            corpus, Path(directory), roles, complete_only=publisher == "elife",
        )
        docs = sorted(files)
        authors = {doc: [] for doc in docs}
        for owner, publications in owners:
            for doc in publications:
                authors[doc].append(owner)
        widths = {doc: (len(people) + 7) // 8 for doc, people in authors.items()}
        costs = {doc: columns.HEADER.size + widths[doc] * cap.bit_count()
                 for doc, cap in caps.items() if cap}
        if sum(costs.values()) != anchor["complete_index_bytes"]:
            raise ValueError("Changed encoded index size")
        budget = anchor["byte_budgets"]["25"]
        traces, reference = cold_workload(dict.fromkeys(asks, owners), state(owners, labels, caps, set()), labels)
        urls = {doc: f"/source/{i}" for i, doc in enumerate(docs)}
        docs_by_url = {href: files[doc] for doc, href in urls.items()}
        ids = {doc: i for i, doc in enumerate(docs)}
        plans, requests = {}, []
        for fold in FOLDS:
            mask = 1 << fold
            train = [ask for ask in asks if not ask & mask]
            held = [ask for ask in asks if ask & mask]
            for arm, selected in columns.designs(owners, caps, widths, costs, train, traces, budget).items():
                if arm in ARMS:
                    payload = columns.encode(ids, authors, labels, selected)
                    if len(payload) != anchor["folds"][str(fold)]["25"][arm]["payload_bytes"]:
                        raise ValueError("Changed packed index contents")
                    plans[f"/{fold}/{arm}"] = columns.decode(payload, docs, authors, caps)
            requests.extend(bodies(roles, held, reference, fold))

        Random(20261121).shuffle(requests)
        arm_order = ARMS[order_offset:] + ARMS[:order_offset]
        for clients, delay in PROFILES:
            profile = {}
            with serve(SourceHandler, files=docs_by_url, delay=delay / 1000) as source:
                with serve(SearchHandler, source_port=source, plans=plans, roles=roles,
                           owners=owners, caps=caps, source_urls=urls) as search:
                    for arm in arm_order:
                        report = execute(search, requests, arm, clients)
                        expected_reads = sum(anchor["folds"][str(fold)]["25"][arm]["file_reads"] for fold in FOLDS)
                        expected_bytes = sum(anchor["folds"][str(fold)]["25"][arm]["decoded_bytes"] for fold in FOLDS)
                        if (report["source_reads"], report["source_payload_bytes"]) != (expected_reads, expected_bytes):
                            raise AssertionError("HTTP fetch volume did not reproduce the warmed local path")
                        profile[arm] = report
            results[f"{clients}_clients_{delay}_ms"] = profile
            print(f"{publisher}: measured {clients} clients, {delay} ms added source delay", flush=True)

    write_json(output, dict(
        publisher=publisher, profiles=results, byte_cap=budget, order_offset=order_offset,
        fixture_sha256=digest(corpus), prior_sha256=digest(prior), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "prior", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", choices=("plos", "elife"), required=True)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(**vars(parser.parse_args()))
