"""Replay offered load for a single GPU request worker with fixed-width encodings."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding import TASK
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


WIDTH = 64
RATES = (60, 120, 240)
ARMS = {"serial": (1, 0.0, 0), "batch4": (4, .002, 0), "batch8": (8, .002, 0), "padded8": (8, .002, 8)}


@dataclass(frozen=True)
class Request:
    panel: str
    payload: str


@torch.inference_mode()
def encode(model, topics):
    strings = [f"Instruct: {TASK}\nQuery:{topic}" for topic in topics]
    tokens = model.tokenizer(strings, padding="max_length", max_length=WIDTH, return_tensors="pt").to("cuda")
    if tokens.input_ids.shape != (len(topics), WIDTH):
        raise ValueError("Unexpected query width")
    tokens["position_ids"] = (tokens.attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    hidden = model.model(**tokens).last_hidden_state[:, -1, :].float()
    return torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()


def execute(indexes, model, requests, pad_to=0):
    atoms = [tuple((item["topic"], item["role"]) for item in json.loads(req.payload)["requirements"])
             for req in requests]
    padded = [*atoms, *([atoms[0]] * max(0, pad_to - len(atoms)))]
    encoded = encode(model, [topic for conditions in padded for topic, _ in conditions])
    for i, (req, conditions) in enumerate(zip(requests, atoms, strict=True)):
        index = indexes[req.panel]
        vectors = encoded[2*i:2*i+2]
        index.encoder = SimpleNamespace(encode=lambda topics, query: vectors)
        ranks = index.rank([topic for topic, _ in conditions])
        plan, records = live.inspect(index, conditions, ranks, "joint")
        if len(records) != 3:
            raise AssertionError("The joint path must open exactly three declarations")
        yield live.render(records, plan, conditions, ranks), perf_counter()


def delay_until(deadline):
    remaining = deadline - perf_counter()
    if remaining > 0:
        sleep(remaining)


def measure(indexes, model, trace, arrivals, canonical, fixed_shape, arm):
    capacity, hold, pad_to = ARMS[arm]
    latencies, queue, service = [], [], []
    counts = Counter()
    changed = changed_fixed = 0
    started = perf_counter()
    cursor = 0
    latest = started
    while cursor < len(trace):
        delay_until(started + arrivals[cursor])
        now = perf_counter()
        end = min(cursor + capacity, len(trace))
        # Use the first arrival's absolute hold deadline. Do not delay
        # queued work by an additional timer after an earlier batch.
        deadline = started + arrivals[cursor] + hold
        if capacity > 1 and now < deadline:
            delay_until(min(deadline, started + arrivals[end-1]))
        dispatched = perf_counter()
        next_cursor = cursor + 1
        while next_cursor < end and started + arrivals[next_cursor] <= dispatched:
            next_cursor += 1
        counts[next_cursor-cursor] += 1
        group = trace[cursor:next_cursor]
        for i, (response, completed) in enumerate(execute(indexes, model, group, pad_to), cursor):
            enqueued = started + arrivals[i]
            latencies.append((completed-enqueued) * 1000)
            queue.append((dispatched-enqueued) * 1000)
            service.append((completed-dispatched) * 1000)
            changed += response != canonical[trace[i]]
            changed_fixed += response != fixed_shape[trace[i]]
            latest = completed
        cursor = next_cursor
    return dict(
        requests=len(trace), changed_responses=changed, changed_vs_padded_reference=changed_fixed,
        trace_rps=(len(trace)-1)/arrivals[-1], achieved_rps=len(trace)/(latest-started),
        p50_ms=float(np.quantile(latencies, .50)), p95_ms=float(np.quantile(latencies, .95)),
        queue_p95_ms=float(np.quantile(queue, .95)), service_p95_ms=float(np.quantile(service, .95)),
        batch_sizes=dict(sorted(counts.items())),
    )


def trace_for(requests, rps):
    rng = Random(20261120)
    order = list(requests)
    rng.shuffle(order)
    at = 0.
    arrival = []
    for _ in order:
        arrival.append(at)
        at += rng.expovariate(rps)
    return order, arrival


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "output"):
        parser.add_argument(path, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result on Slurm CUDA")
    checkpoint = live.verify_embedding(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    plos, _ = live.BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    plos_subjects = live.BASE.publisher_topics([args.plos_xml])
    life_subjects, _, life_requests = live.elife.build_requests(life)["topics"]
    by_panel = {
        "plos": (plos, plos_subjects, live.BASE.query_panel(live.BASE.annotation_postings(plos, plos_subjects))),
        "elife": (life, life_subjects, life_requests),
    }
    with TemporaryDirectory() as root:
        indexes = {panel: live.LocalIndex(catalogue, encoder, Path(root) / panel)
                   for panel, (catalogue, _, _) in by_panel.items()}
        requests = [Request(panel, live.request(atoms)) for panel, (_, _, rows) in by_panel.items() for atoms in rows]
        canonical = {req: next(execute(indexes, encoder, [req]))[0] for req in requests}
        fixed_shape = {req: next(execute(indexes, encoder, [req], 8))[0] for req in requests}
        known = Counter()
        for panel, (_, subjects, queries) in by_panel.items():
            for atoms in queries:
                matches = json.loads(canonical[Request(panel, live.request(atoms))])["matches"]
                known[panel] += int(bool(matches) and all(t in subjects[matches[0][i+1]]
                                                          for i, (t, _) in enumerate(atoms)))
        results = {}
        for rate in RATES:
            trace, arrivals = trace_for(requests, rate)
            results[str(rate)] = {}
            keys = tuple(ARMS)
            offset = args.order_offset % len(keys)
            for arm in keys[offset:] + keys[:offset]:
                # Warm execution shapes without applying a response cache.
                for size in range(1, ARMS[arm][0] + 1):
                    list(execute(indexes, encoder, trace[:size], ARMS[arm][2]))
                results[str(rate)][arm] = measure(indexes, encoder, trace, arrivals, canonical, fixed_shape, arm)
            print(f"Completed offered load {rate}", flush=True)
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), results=results, canonical_known=dict(known),
        padded_reference_changes=sum(canonical[req] != fixed_shape[req] for req in requests),
        input_width=WIDTH, order_offset=args.order_offset, checkpoint=checkpoint,
        elife_projection=checksum, script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
