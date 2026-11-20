"""Measure live GPU retrieval followed by equal-output HTTP source prechecks."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_prefix_completion as prefix
from experiments.credit import joint_request_queue as batches
from experiments.credit import live_joint_source_request as live
from experiments.credit import prefix_source_http as source_http
from experiments.followir.core17_reranking import digest, write_json


POLICIES = ("parallel_three", "eager_topology")
BATCH_SIZE, HOLD_SECONDS, PAD_TO = batches.ARMS["padded8"]
RATES = (60, 165)
REPEATS = 2


def plan_batch(indexes, urls, model, requests):
    atoms = [tuple((item["topic"], item["role"]) for item in json.loads(req.payload)["requirements"])
             for req in requests]
    padded = atoms + [atoms[0]] * (PAD_TO - len(atoms))
    encoded = batches.encode(model, [topic for pair in padded for topic, _ in pair])
    planned = []
    for i, (request, conditions) in enumerate(zip(requests, atoms, strict=True)):
        index = indexes[request.panel]
        vectors = encoded[2*i:2*i+2]
        index.encoder = SimpleNamespace(encode=lambda topics, query: vectors)
        ranks = index.rank([topic for topic, _ in conditions])
        plan = prefix.nominate(index, ranks)
        byline = index.catalogue.byline
        planned.append(dict(
            atoms=conditions, ranks=ranks, plan=plan, byline=byline, urls=urls[request.panel],
            topology=source_http.third_can_participate(plan, ranks, byline),
        ))
    return planned


async def planner(pending, indexes, urls, model, executor, sizes):
    loop = asyncio.get_running_loop()
    while (first := await pending.get()) is not None:
        group = [first]
        deadline = first[2] + HOLD_SECONDS
        while len(group) < BATCH_SIZE:
            if not pending.empty():
                group.append(pending.get_nowait())
            else:
                time_left = deadline - loop.time()
                if time_left <= 0:
                    break
                try:
                    group.append(await asyncio.wait_for(pending.get(), time_left))
                except TimeoutError:
                    break
        sizes[len(group)] += 1
        start = loop.time()
        try:
            plans = await loop.run_in_executor(
                executor, plan_batch, indexes, urls, model, [item[0] for item in group],
            )
            end = loop.time()
            for (_, future, _), plan in zip(group, plans, strict=True):
                future.set_result((plan, start, end))
        except Exception as error:
            for _, future, _ in group:
                if not future.done():
                    future.set_exception(error)
            raise


async def replay(indexes, urls, model, references, paths, trace, rate, policy, executor):
    loop = asyncio.get_running_loop()
    service = source_http.SourceService(paths)
    listener = await asyncio.start_server(service.handle, "127.0.0.1", 0, backlog=512)
    port = listener.sockets[0].getsockname()[1]
    pending = asyncio.Queue()
    sizes = Counter()
    owner = asyncio.create_task(planner(pending, indexes, urls, model, executor, sizes))

    async def request(scheduled, case):
        await asyncio.sleep(max(0, scheduled - loop.time()))
        future = loop.create_future()
        pending.put_nowait((case, future, scheduled))
        plan, dispatched, ready = await future
        answer, reads = await source_http.resolve(port, plan, policy)
        finished = loop.time()
        if answer != references[case]:
            raise AssertionError("Live request differs from the equal-shape reference response")
        return reads, (dispatched-scheduled)*1000, (ready-dispatched)*1000, (finished-ready)*1000, (finished-scheduled)*1000

    async with listener:
        try:
            await request(loop.time(), trace[0])
            service.completed = service.bytes = 0
            sizes.clear()
            start = loop.time() + .10
            rows = await asyncio.wait_for(asyncio.gather(*(
                request(start + i/rate, case) for i, case in enumerate(trace)
            )), timeout=source_http.REQUEST_TIMEOUT_SECONDS + len(trace)/rate)
            elapsed = loop.time() - start
        finally:
            pending.put_nowait(None)
            await owner

    if sum(row[0] for row in rows) != service.completed or sum(k*v for k,v in sizes.items()) != len(rows):
        raise AssertionError("Incomplete source or GPU batch counters")
    stage_ms = {
        name: float(np.quantile([row[i] for row in rows], .95))
        for i, name in enumerate(("planner_queue", "gpu_and_retrieval", "source_and_match", "total"), 1)
    }
    return dict(
        requests=len(rows), equal_responses=len(rows), source_reads=service.completed,
        source_payload_bytes=service.bytes, achieved_rps=len(rows)/elapsed,
        batch_sizes=dict(sorted(sizes.items())), p95_ms=stage_ms,
    )


async def measure(indexes, urls, model, references, paths, trace, offset):
    with ThreadPoolExecutor(max_workers=1) as worker:
        results = {}
        order = POLICIES[offset:] + POLICIES[:offset]
        for rate in RATES:
            results[str(rate)] = {}
            for policy in order:
                report = await replay(indexes, urls, model, references, paths, trace, rate, policy, worker)
                results[str(rate)][policy] = report
                print(f"{rate} {policy}: {report['source_reads']} reads; P95 {report['p95_ms']['total']:.2f} ms",
                      flush=True)
        return results


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
    life, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or checksum != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"] or live.DEPTH != expected["query_depth"]
    ):
        raise ValueError("Frozen inputs differ from the request reference")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_cases = base.query_panel(base.annotation_postings(plos, plos_subjects))
    _, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_cases), elife=(life, life_cases))
    model = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory() as scratch:
        indexes = {name: live.LocalIndex(catalogue, model, Path(scratch) / name)
                   for name, (catalogue, _) in panels.items()}
        urls = {name: {doi: f"/{name}/{i}" for i, doi in enumerate(index.ids)}
                for name, index in indexes.items()}
        paths = {route: indexes[name].paths[doi] for name, table in urls.items() for doi, route in table.items()}
        cases = [batches.Request(name, live.request(atoms))
                 for name, (_, queries) in panels.items() for atoms in queries]
        references = {}
        reachable = Counter()
        for case in cases:
            item, = plan_batch(indexes, urls, model, [case])
            index = indexes[case.panel]
            loaded = index.open_sources(item["plan"])
            proofs = live.BASE.choose_proofs(prefix.evidence(loaded, item["atoms"], item["ranks"]), item["ranks"])
            references[case] = source_http.serialize(next(iter(proofs), None))
            reachable[case.panel] += int(item["topology"])
        trace = []
        for repeat in range(REPEATS):
            shuffled = list(cases)
            Random(20261120 + repeat).shuffle(shuffled)
            trace.extend(shuffled)
        results = asyncio.run(measure(indexes, urls, model, references, paths, trace, args.order_offset))
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), order_offset=args.order_offset, checkpoint=checkpoint,
        reads_with_topological_links=dict(reachable),
        input_width=batches.WIDTH, maximum_batch_requests=BATCH_SIZE, padded_batch_requests=PAD_TO,
        hold_ms=HOLD_SECONDS*1000, source_workers=source_http.SOURCE_WORKERS,
        source_delay_ms=source_http.READ_DELAY_SECONDS*1000, distinct_requests=len(cases),
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)), results=results,
    ))


if __name__ == "__main__":
    main()
