"""Measure HTTP source-access tradeoffs with frozen joint nominations."""

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_prefix_completion as prefix
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


POLICIES = ("parallel_three", "eager_topology", "certified_prefix")
RATES = (60, 165)
REPEATS = 2
SOURCE_WORKERS = 6
READ_DELAY_SECONDS = .016
REQUEST_TIMEOUT_SECONDS = 12


def third_can_participate(plan, ranks, byline):
    third = plan[2]
    return any(
        anchor in ranks[side] and third in ranks[1 - side]
        and not byline[anchor].isdisjoint(byline[third])
        for anchor in plan[:2] for side in range(2)
    )


def load_panel(catalogue, encoder, directory, cases, panel):
    index = live.LocalIndex(catalogue, encoder, directory)
    paths = {f"/{panel}/{i}": index.paths[doi] for i, doi in enumerate(index.ids)}
    urls = {doi: url for doi, url in zip(index.ids, paths, strict=True)}
    requests = []
    for atoms in cases:
        ranks = index.rank([topic for topic, _ in atoms])
        plan, proof = prefix.fixed(index, atoms, ranks)
        byline = catalogue.byline
        requests.append(dict(atoms=atoms, ranks=ranks, plan=plan, urls=urls, byline=byline,
                             topology=third_can_participate(plan, ranks, byline),
                             reference=serialize(proof), panel=panel))
    return paths, requests


def serialize(proof):
    return json.dumps({"matches": [proof] if proof else []}, separators=(",", ":")).encode()


class SourceService:
    def __init__(self, paths):
        self.paths = paths
        self.slots = asyncio.Semaphore(SOURCE_WORKERS)
        self.completed = 0
        self.bytes = 0

    async def handle(self, reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            verb, target, version = header.split(b"\r\n", 1)[0].decode("ascii").split(" ")
            if verb != "GET" or version != "HTTP/1.1" or target not in self.paths:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            async with self.slots:
                await asyncio.sleep(READ_DELAY_SECONDS)
                payload = self.paths[target].read_bytes()
                header = f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                writer.write(header.encode() + payload)
                await writer.drain()
                self.completed += 1
                self.bytes += len(payload)
        finally:
            writer.close()
            await writer.wait_closed()


async def fetch(port, url):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(f"GET {url} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        header = await reader.readuntil(b"\r\n\r\n")
        if not header.startswith(b"HTTP/1.1 200 OK\r\n"):
            raise RuntimeError("HTTP source read failed")
        fields = dict(line.split(b": ", 1) for line in header.split(b"\r\n")[1:-2])
        return {"authors": json.loads(await reader.readexactly(int(fields[b"Content-Length"])))}
    finally:
        writer.close()
        await writer.wait_closed()


async def resolve(port, request, policy):
    atoms, ranks, plan = (request[name] for name in ("atoms", "ranks", "plan"))
    eager_third = policy == "parallel_three" or (policy == "eager_topology" and request["topology"])
    docs = plan if eager_third else plan[:2]
    loaded = dict(zip(docs, await asyncio.gather(*(fetch(port, request["urls"][doi]) for doi in docs)), strict=True))
    observed = prefix.evidence(loaded, atoms, ranks)
    proofs = live.BASE.choose_proofs(observed, ranks)
    best = next(iter(proofs), None)
    if policy == "certified_prefix" and request["topology"]:
        limited = tuple({doi: rank for doi, rank in side.items() if doi in plan} for side in ranks)
        viable = prefix.alternative_proofs(observed, limited, request["byline"], plan[:2])
        if viable and (best is None or min(item[0] for item in viable) < prefix.proof_key(best, ranks)):
            newest = await fetch(port, request["urls"][plan[2]])
            loaded[plan[2]] = newest
            proofs = live.BASE.choose_proofs(prefix.evidence(loaded, atoms, ranks), ranks)
            best = next(iter(proofs), None)
    return serialize(best), len(loaded)


async def trial(paths, trace, rate, policy):
    service = SourceService(paths)
    server = await asyncio.start_server(service.handle, "127.0.0.1", 0, backlog=512)
    port = server.sockets[0].getsockname()[1]
    loop = asyncio.get_running_loop()
    async with server:
        for item in trace[:3]:
            expected, _ = await asyncio.wait_for(resolve(port, item, policy), REQUEST_TIMEOUT_SECONDS)
            if expected != item["reference"]:
                raise AssertionError("Warmup answer disagreed")
        service.completed = service.bytes = 0
        start = loop.time() + .10

        async def request(i, item):
            scheduled = start + i / rate
            await asyncio.sleep(max(0, scheduled - loop.time()))
            answer, reads = await asyncio.wait_for(resolve(port, item, policy), REQUEST_TIMEOUT_SECONDS)
            if answer != item["reference"]:
                raise AssertionError("Source access policy changed an expert or citation")
            return item["panel"], reads, (loop.time() - scheduled) * 1000

        rows = await asyncio.gather(*(request(i, item) for i, item in enumerate(trace)))
        elapsed = loop.time() - start
        if sum(row[1] for row in rows) != service.completed:
            raise AssertionError("Client reads differ from service counters")
        panel_reads = Counter()
        for panel, count, _ in rows:
            panel_reads[panel] += count
        timing = [ms for _, _, ms in rows]
        return dict(
            requests=len(rows), equal_responses=len(rows), source_reads=service.completed,
            source_payload_bytes=service.bytes, panel_reads=dict(panel_reads),
            drained_rps=len(rows) / elapsed, p50_ms=float(np.quantile(timing, .50)),
            p95_ms=float(np.quantile(timing, .95)),
            three_read_p95_ms=float(np.quantile([ms for _, count, ms in rows if count == 3], .95))
            if any(count == 3 for _, count, _ in rows) else None,
        )


async def replay(paths, cases, offset):
    trace = []
    for repeat in range(REPEATS):
        order = list(cases)
        Random(20261120 + repeat).shuffle(order)
        trace.extend(order)
    out = {}
    for rate in RATES:
        order = POLICIES[offset:] + POLICIES[:offset]
        out[str(rate)] = {}
        for policy in order:
            score = await trial(paths, trace, rate, policy)
            out[str(rate)][policy] = score
            print(f"{rate} {policy}: {score['source_reads']} reads, P95 {score['p95_ms']:.2f} ms", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"] or live.DEPTH != expected["query_depth"]
    ):
        raise ValueError("Frozen inputs differ from the request reference")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_cases = base.query_panel(base.annotation_postings(plos, plos_subjects))
    _, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_cases), elife=(life, life_cases))
    encoder = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory() as scratch:
        sources, requests, topology = {}, [], {}
        for name, (catalogue, cases) in panels.items():
            paths, plans = load_panel(catalogue, encoder, Path(scratch) / name, cases, name)
            if len(plans) != expected["panels"][name]["queries"] or sum(
                bool(json.loads(p["reference"])["matches"]) for p in plans
            ) != expected["panels"][name]["arms"]["structured"]["joint"]["issued"]:
                raise ValueError(f"Reference issuance changed for {name}")
            sources.update(paths)
            requests.extend(plans)
            topology[name] = dict(cases=len(plans), no_third_link=sum(not row["topology"] for row in plans))
        results = asyncio.run(replay(sources, requests, args.order_offset))
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), fixed_pool_depth=live.DEPTH, repeats=REPEATS,
        source_workers=SOURCE_WORKERS, added_source_delay_ms=READ_DELAY_SECONDS*1000,
        topology=topology, results=results, order_offset=args.order_offset, embedding_checkpoint=checkpoint,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
