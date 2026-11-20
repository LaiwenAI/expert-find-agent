"""Measure remote declaration access for the exact structured expert path."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from hashlib import sha256
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
from experiments.credit.role_index_read_waves import frontier_rank
from experiments.credit.budgeted_declaration_index import FOLDS, cold_workload, state
from experiments.credit.declaration_index_publisher_transfer import inventory
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.cof_loopback_http import Server


ARMS = ("whole_author", "whole_cold", "column_smoothed")
PROFILES = ((1, 0), (4, 0), (4, 2))
WAVES = {"owner_serial": 0, "frontier_1": 1, "frontier_2": 2, "frontier_4": 4}
ADAPTIVE = "frontier_adaptive"
WAVE_COST_MS = 0.75


def choose_parallel_width(docs, delays):
    expected = [delays[doc] for doc in docs]
    overlappable = sum(expected) - max(expected)
    return len(docs) if overlappable >= WAVE_COST_MS else 1


def source_delays(docs, fraction, seed):
    """Assign the experimental delay independently of query and proof results."""
    slow = set(Random(seed).sample(docs, round(len(docs) * fraction)))
    return {doc: (2.0 if doc in slow else 0.0) for doc in docs}


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
        seconds = self.server.delay_by_url.get(self.path, self.server.delay)
        if seconds:
            sleep(seconds)
        self.respond(file.read_bytes())


class SearchHandler(Handler):
    def setup(self):
        super().setup()
        self.sources = [HTTPConnection("127.0.0.1", self.server.source_port, timeout=10)]
        self.pool = None

    def finish(self):
        if self.pool is not None:
            self.pool.shutdown()
        for connection in self.sources:
            connection.close()
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

        sizes = {}
        arm = self.path.rsplit("/", 1)[-1]
        width = self.server.widths.get(arm, 0)

        def fetch(connection, doc):
            payload, _ = receive(connection, "GET", self.server.source_urls[doc])
            return json.loads(payload), len(payload)

        def load(doc):
            value, sizes[doc] = fetch(self.sources[0], doc)
            return value

        def load_many(docs):
            if len(docs) == 1:
                return {docs[0]: load(docs[0])}
            if self.pool is None:
                self.pool = ThreadPoolExecutor(max_workers=width)
            while len(self.sources) < len(docs):
                self.sources.append(HTTPConnection("127.0.0.1", self.server.source_port, timeout=10))
            fetched = self.pool.map(fetch, self.sources[:len(docs)], docs)
            declarations = {}
            for doc, (value, size) in zip(docs, fetched, strict=True):
                declarations[doc] = value
                sizes[doc] = size
            return declarations

        if width:
            choose_width = None
            if arm == ADAPTIVE:
                def choose_width(docs):
                    return choose_parallel_width(docs, self.server.expected_delays)
            matches, reads, rounds = frontier_rank(
                self.server.owners, self.server.caps, plan, ask, load_many, width,
                choose_width=choose_width,
            )
        else:
            matches, reads = columns.partial_rank(self.server.owners, self.server.caps, plan, ask, load)
            rounds = reads
        payload = json.dumps({"matches": matches}, separators=(",", ":")).encode()
        headers = {"X-Source-Reads": reads, "X-Source-Bytes": sum(sizes.values())}
        if self.server.widths:
            source_set = json.dumps(sorted(sizes), separators=(",", ":")).encode()
            headers.update({"X-Source-Rounds": rounds, "X-Source-Set": sha256(source_set).hexdigest()})
        self.respond(payload, headers)


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


def execute(port, requests, arm, clients, *, inspect_waves=False):
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
                row = (elapsed, int(headers["X-Source-Reads"]), int(headers["X-Source-Bytes"]))
                if inspect_waves:
                    row += (int(headers["X-Source-Rounds"]), headers["X-Source-Set"])
                scores.append(row)
        return scores

    with ThreadPoolExecutor(max_workers=clients) as pool:
        rows = [item for stream in pool.map(replay, [requests[i::clients] for i in range(clients)])
                for item in stream]
    report = dict(
        requests=len(rows), exact_responses=len(rows),
        source_reads=sum(r[1] for r in rows), source_payload_bytes=sum(r[2] for r in rows),
        response_p50_ms=float(np.quantile([r[0] for r in rows], .50)),
        response_p95_ms=float(np.quantile([r[0] for r in rows], .95)),
    )
    if inspect_waves:
        report.update(
            source_rounds=sum(row[3] for row in rows),
            p95_source_rounds=float(np.quantile([row[3] for row in rows], .95)),
            source_sets_sha256=sha256("".join(row[4] for row in rows).encode()).hexdigest(),
        )
    return report


def run(corpus, manifest, prior, scratch, output, publisher, order_offset, waves=None, mixed_delay=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a fresh output on a Slurm compute node")
    anchor = json.loads(prior.read_bytes())["publishers"][publisher]
    expected = (columns.bundles.FIXTURE_SHA256 if publisher == "plos"
                else json.loads(manifest.read_bytes())["projection_sha256"])
    if digest(corpus) != expected or anchor["fixture_sha256"] != expected:
        raise ValueError("Unexpected publisher projection")
    if digest(Path(columns.__file__)) != anchor["script_sha256"]:
        raise ValueError("Index planner changed since the warm replay")
    wave_reference = json.loads(waves.read_bytes())[publisher] if waves else None
    placement = "whole_cold" if publisher == "plos" else "column_smoothed"
    if waves and (wave_reference["fixture_sha256"] != expected or wave_reference["prior_sha256"] != digest(prior)):
        raise ValueError("Changed parallel-read inputs")
    if mixed_delay and not waves:
        raise ValueError("Mixed-delay replay requires the reference wave benchmark")
    wave_widths = dict(WAVES) if waves else {}
    if mixed_delay:
        wave_widths[ADAPTIVE] = 4
    arms = tuple(wave_widths) if waves else ARMS
    placements = (placement,) if waves else ARMS
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
                if arm in placements:
                    payload = columns.encode(ids, authors, labels, selected)
                    if len(payload) != anchor["folds"][str(fold)]["25"][arm]["payload_bytes"]:
                        raise ValueError("Changed packed index contents")
                    compiled = columns.decode(payload, docs, authors, caps)
                    for route in (arms if waves else (arm,)):
                        plans[f"/{fold}/{route}"] = compiled
            requests.extend(bodies(roles, held, reference, fold))

        Random(20261121).shuffle(requests)
        arm_order = (arms[::-1] if order_offset else arms) if waves else arms[order_offset:] + arms[:order_offset]
        profiles = [(f"{clients}_clients_{delay}_ms", clients, delay, {}, {}) for clients, delay in PROFILES]
        if mixed_delay:
            sparse = source_delays(docs, .25, 721)
            dense = source_delays(docs, .75, 721)
            shifted = source_delays(docs, .25, 953)
            profiles = [
                ("sparse_known", 4, 0, sparse, sparse),
                ("dense_known", 4, 0, dense, dense),
                ("shifted_known", 4, 0, shifted, shifted),
                ("shifted_stale", 4, 0, shifted, sparse),
            ]
        for profile_name, clients, delay, actual, predicted in profiles:
            profile = {}
            delay_by_url = {urls[doc]: ms / 1000 for doc, ms in actual.items()}
            with serve(SourceHandler, files=docs_by_url, delay=delay / 1000, delay_by_url=delay_by_url) as source:
                with serve(SearchHandler, source_port=source, plans=plans, roles=roles, owners=owners,
                           caps=caps, source_urls=urls, widths=wave_widths, expected_delays=predicted) as search:
                    for arm in arm_order:
                        report = execute(search, requests, arm, clients, inspect_waves=bool(waves))
                        allocation = placement if waves else arm
                        expected_reads = sum(anchor["folds"][str(fold)]["25"][allocation]["file_reads"] for fold in FOLDS)
                        expected_bytes = sum(anchor["folds"][str(fold)]["25"][allocation]["decoded_bytes"] for fold in FOLDS)
                        if (report["source_reads"], report["source_payload_bytes"]) != (expected_reads, expected_bytes):
                            raise AssertionError("HTTP fetch volume did not reproduce the warmed local path")
                        if waves and arm != ADAPTIVE:
                            reference_rounds = wave_reference["results"][placement][arm]["source_rounds"]
                            if report["source_rounds"] != reference_rounds:
                                raise AssertionError("HTTP changed the number of source read waves")
                        profile[arm] = report
                    if waves and len({row["source_sets_sha256"] for row in profile.values()}) != 1:
                        raise AssertionError("HTTP changed the source set for an individual request")
            results[profile_name] = profile
            print(f"{publisher}: measured {profile_name} with {clients} clients", flush=True)

    result = dict(
        publisher=publisher, profiles=results, byte_cap=budget, order_offset=order_offset,
        fixture_sha256=digest(corpus), prior_sha256=digest(prior), script_sha256=digest(Path(__file__)),
    )
    if waves:
        result.update(placement=placement, waves_sha256=digest(waves))
    if mixed_delay:
        result["parallel_overhead_ms"] = WAVE_COST_MS
    write_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "prior", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", choices=("plos", "elife"), required=True)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    parser.add_argument("--waves", type=Path, help="Use exact-source read wave counts as a reference")
    parser.add_argument("--mixed-delay", action="store_true", help="Compare known and stale source-delay metadata")
    run(**vars(parser.parse_args()))
