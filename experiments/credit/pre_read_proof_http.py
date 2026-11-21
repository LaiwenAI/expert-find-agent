"""Replay exact source access over a bounded localhost HTTP service."""

import argparse
import asyncio
from hashlib import sha256
import json
import os
import resource
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from experiments.credit import pre_read_proof_bound as bound
from experiments.credit.prefix_source_http import (
    READ_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SOURCE_WORKERS,
    SourceService,
    fetch,
    serialize,
    third_can_participate,
)


ARMS = ("full", "two_only", "byline", "pre_read")
RATES = (60, 165, 205)
VISIBILITIES = (100, 50)
DEPTH = 128


def prepare(catalogue, model, directory, name):
    linked = bound.linked
    index = linked.replay.live.LocalIndex(catalogue, linked.previous.mixed.QwenTorchEmbedding(model), directory)
    postings, _ = linked.certificate.indexes(catalogue)
    packed = linked.PackedLinks(postings["full"])
    tags = {doc: {" ".join(topic.casefold().split()) for topic in item["topics"]}
            for doc, item in catalogue.ownership.items()}
    base = linked.previous.live.BASE
    requests = base.query_panel(base.annotation_postings(catalogue, tags))
    hidden = {doc for doc in index.ids if int(sha256(doc.encode()).hexdigest()[:8], 16) % 2}
    urls = {doc: f"/{name}/{i}" for i, doc in enumerate(index.ids)}
    paths = {urls[doc]: index.paths[doc] for doc in index.ids}
    cases = {rate: [] for rate in VISIBILITIES}
    for atoms in requests:
        ranks = index.rank([topic for topic, _ in atoms], depth=DEPTH)
        anchors, extra = linked.replay.live.simple_order(ranks, depth=DEPTH)
        complete = tuple(packed.facet(lane, role) for lane, (_, role) in zip(ranks, atoms, strict=True))
        for visibility in VISIBILITIES:
            role_views = complete if visibility == 100 else tuple(
                {doc: (set(), packed.compiled.eligible[doc]) if doc in hidden else view
                 for doc, view in lane.items()} for lane in complete)
            plan = linked.replay.typed.source_plan(ranks, role_views, [*anchors, extra[0]], DEPTH)
            reference, _ = bound.verify(index.open_sources(plan), atoms, ranks)
            facets = tuple({doc: lane[doc] for doc in plan if doc in lane} for lane in role_views)
            cases[visibility].append(dict(
                atoms=atoms, ranks=ranks, plan=plan, facets=facets, urls=urls,
                byline_third=third_can_participate(plan, ranks, catalogue.byline),
                reference=serialize(reference),
            ))
    return paths, cases


def choose(case, arm):
    plan = case["plan"]
    if arm == "full" or arm == "byline" and case["byline_third"]:
        return plan, None
    if arm != "pre_read":
        return plan[:2], None
    ranks, facets = case["ranks"], case["facets"]
    upper = bound.best_bound(plan, ranks, facets, True)
    if upper is None:
        return [], None
    if upper == bound.best_bound(plan, ranks, facets, False):
        return list(upper[1:]), upper
    return plan, None


async def fetch_docs(port, case, docs):
    values = await asyncio.gather(*(fetch(port, case["urls"][doc]) for doc in docs))
    return dict(zip(docs, values, strict=True))


async def resolve(port, case, arm):
    docs, certified = choose(case, arm)
    if not docs:
        return serialize(None), 0, 0
    loaded = await fetch_docs(port, case, docs)
    match, _ = bound.verify(loaded, case["atoms"], case["ranks"])
    if certified is None or match == certified:
        return serialize(match), len(loaded), 1
    missing = [doc for doc in case["plan"] if doc not in loaded]
    loaded.update(await fetch_docs(port, case, missing))
    match, _ = bound.verify(loaded, case["atoms"], case["ranks"])
    return serialize(match), len(loaded), 2


async def trial(paths, trace, rate, arm):
    service = SourceService(paths)
    server = await asyncio.start_server(service.handle, "127.0.0.1", 0, backlog=512)
    port = server.sockets[0].getsockname()[1]
    loop = asyncio.get_running_loop()

    async def checked(case):
        answer, reads, waves = await asyncio.wait_for(resolve(port, case, arm), REQUEST_TIMEOUT_SECONDS)
        if answer != case["reference"]:
            raise AssertionError(f"{arm} changed a fixed-plan response")
        return reads, waves

    async with server:
        for case in trace[:3]:
            await checked(case)
        service.completed = service.bytes = 0
        start = loop.time() + .10

        async def arrival(i, case):
            scheduled = start + i / rate
            await asyncio.sleep(max(0, scheduled - loop.time()))
            submitted = loop.time()
            reads, waves = await checked(case)
            return reads, waves, (loop.time() - scheduled) * 1000, (submitted - scheduled) * 1000

        rows = await asyncio.gather(*(arrival(i, case) for i, case in enumerate(trace)))
        elapsed = loop.time() - start
    reads, waves, milliseconds, lag = zip(*rows, strict=True)
    if sum(reads) != service.completed:
        raise AssertionError("Client reads and server completions differ")
    with_reads = [ms for count, ms in zip(reads, milliseconds, strict=True) if count]
    return dict(
        matched_responses=len(rows), reads=service.completed, waves=sum(waves), payload_bytes=service.bytes,
        zero_read_requests=reads.count(0), drained_rps=len(rows) / elapsed,
        p50_ms=float(np.quantile(milliseconds, .5)), p95_ms=float(np.quantile(milliseconds, .95)),
        p95_when_reading_ms=float(np.quantile(with_reads, .95)), release_lag_p95_ms=float(np.quantile(lag, .95)),
        final_drain_ms=(elapsed - (len(rows) - 1) / rate) * 1000,
    )


async def replay(paths, cases, expected, reverse):
    out = {}
    for visibility in (VISIBILITIES[::-1] if reverse else VISIBILITIES):
        panel = cases[visibility]
        seeded = list(panel)
        Random(20261121).shuffle(seeded)
        trace = list(reversed(seeded)) if reverse else seeded
        out[visibility] = {}
        for rate in (RATES[::-1] if reverse else RATES):
            metrics = {}
            for arm in (ARMS[::-1] if reverse else ARMS):
                row = await trial(paths, trace, rate, arm)
                if arm in ("full", "two_only", "pre_read"):
                    if row["reads"] != expected[visibility, arm]:
                        raise ValueError("Service access count differs from the archived local replay")
                metrics[arm] = row
                print(f"{visibility}% {rate}/s {arm}: {row['reads']} reads, P95 {row['p95_ms']:.2f} ms",
                      flush=True)
            out[visibility][rate] = metrics
    return out


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU allocation")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 8192:
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
    linked = bound.linked
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = linked.previous.live.verify_embedding(args.embedding)
    if checkpoint != reference["embedding"]:
        raise ValueError("Changed embedding checkpoint")
    counts = {(visibility, arm): sum(
        reference["reports"][f"{name}_{DEPTH}"]["panels"][str(visibility)][arm]["reads"]
        for name in linked.PANELS) for visibility in VISIBILITIES for arm in ("full", "two_only", "pre_read")}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        paths, cases = {}, {visibility: [] for visibility in VISIBILITIES}
        started = perf_counter()
        for name in linked.PANELS:
            projection = getattr(args, name)
            manifest = getattr(args, name + "_manifest")
            catalogue = linked.load(projection, manifest)
            fresh, requests = prepare(catalogue, args.embedding, Path(scratch) / name, name)
            for visibility, panel in requests.items():
                count = reference["reports"][f"{name}_{DEPTH}"]["panels"][str(visibility)]["full"]["issued"]
                if sum(case["reference"] != serialize(None) for case in panel) != count:
                    raise ValueError("Full plan issuance does not reproduce")
                cases[visibility].extend(panel)
            paths.update(fresh)
        prepare_seconds = perf_counter() - started
        reports = asyncio.run(replay(paths, cases, counts, args.reverse))
    linked.prior.write_json(args.output, dict(
        file_descriptor_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        reports=reports, job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
        gpu=torch.cuda.get_device_name(), reference_sha256=linked.prior.digest(args.reference),
        script_sha256=linked.prior.digest(Path(__file__)), depth=DEPTH,
        added_source_delay_ms=READ_DELAY_SECONDS * 1000, worker_slots=SOURCE_WORKERS,
        prepare_seconds=prepare_seconds,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("extended", "extended_manifest", "confirmation", "confirmation_manifest", "embedding",
                  "reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
