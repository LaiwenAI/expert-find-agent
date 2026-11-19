"""Benchmark frozen contribution verification with open-loop arrivals and GPU batching."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter, sleep

import numpy as np
import torch

from experiments.credit import compile_legacy_notes as indexed
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_note_cascade as cascade


COLD_RESULTS = "8f9b302970fa702cd6b1a9958e1b09ed1e3702ace013c5ada426ee407d2468ab"
WAIT = .002
RECHECK_BAND = .25
RATES = (20, 55, 90)
TRIALS = 3
ARMS = {
    "pair_reader": (False, 1),
    "pair_character": (True, 1),
    "batch_reader": (False, 8),
    "batch_character": (True, 8),
}


@dataclass
class Request:
    number: int
    row: tuple
    ready: float
    decision: np.ndarray
    pending: np.ndarray


def trace(rows, rate, seed):
    rng = np.random.default_rng(seed)
    order = [int(i) for _ in range(2) for i in rng.permutation(len(rows))]
    offsets = np.r_[0., rng.exponential(1 / rate, len(order) - 1).cumsum()]
    return order, offsets


def logits(requests, engines):
    model, tokenizer, digits = engines
    contexts, flips = zip(*(cascade.reader_contexts(row, 1) for row in requests), strict=True)
    margins = indexed.judge(model, tokenizer, digits, "Apply the decision separately to this author.",
                            [doc for pair in contexts for doc in pair])
    return np.stack([margins[2 * i:2 * i + 2][::-1] if flip else margins[2 * i:2 * i + 2]
                     for i, flip in enumerate(flips)])


def worker_batch(first, jobs, max_pairs):
    group = [first]
    done = False
    if max_pairs <= 1:
        return group, done
    deadline = first.ready + WAIT
    while len(group) < max_pairs:
        try:
            following = jobs.get(timeout=max(0., deadline - perf_counter()))
        except Empty:
            break
        if following is None:
            done = True
            break
        group.append(following)
    return group, done


def trial(rows, catalog, engines, gate, cutoff, rank_order, offsets, max_pairs):
    scheduled = perf_counter() + .030 + offsets
    jobs = Queue()
    decisions, finish, producer_lag = {}, {}, {}
    tally, sizes, waits = Counter(), [], []

    def produce():
        try:
            for number, (index, due) in enumerate(zip(rank_order, scheduled, strict=True)):
                ahead = due - perf_counter()
                if ahead > 0.:
                    sleep(ahead)
                producer_lag[number] = max(0., perf_counter() - due)
                row = rows[index]
                if gate is None:
                    chosen = np.zeros(2, dtype=bool)
                    pending = np.ones(2, dtype=bool)
                else:
                    result = cascade.score_pair(row, catalog[row[0]], 1, None, gate.features)
                    chosen, pending = cascade.first_stage(result, gate)
                if pending.any():
                    jobs.put(Request(number, row, perf_counter(), chosen, pending))
                else:
                    decisions[number] = tuple(bool(x) for x in chosen)
                    finish[number] = perf_counter()
        finally:
            jobs.put(None)

    with ThreadPoolExecutor(max_workers=1) as pool:
        producer = pool.submit(produce)
        done = False
        while not done:
            first = jobs.get()
            if first is None:
                break
            batch, done = worker_batch(first, jobs, max_pairs)
            started = perf_counter()
            waits.extend(started - request.ready for request in batch)
            values = logits([request.row for request in batch], engines)
            tally["regular_gpu_batches"] += 1
            tally["reader_pairs"] += len(batch)
            sizes.append(len(batch))
            for i, req in enumerate(batch):
                if len(batch) > 1 and (np.abs(values[i][req.pending] - cutoff) <= RECHECK_BAND).any():
                    values[i] = logits([req.row], engines)[0]
                    tally["singleton_rechecks"] += 1
                decision = req.decision | (req.pending & (values[i] > cutoff))
                decisions[req.number] = tuple(bool(x) for x in decision)
                finish[req.number] = perf_counter()
        producer.result()
    n = len(scheduled)
    if set(finish) != set(range(n)) or decisions.keys() != finish.keys():
        raise AssertionError("Missing response during open-loop replay")
    elapsed = np.asarray([finish[i] - scheduled[i] for i in range(n)])
    output = [decisions[i] for i in range(n)]
    return dict(tally, latency_p50_p95_p99_ms=(1000 * np.quantile(elapsed, [.5, .95, .99])).tolist(),
                waiting_p95_ms=float(np.quantile(waits, .95)) * 1000 if waits else 0.,
                source_to_completion_rps=n / (max(finish.values()) - scheduled[0]),
                scheduled_rps=n / (scheduled[-1] - scheduled[0]),
                producer_lag_p95_ms=1000 * float(np.quantile(list(producer_lag.values()), .95)),
                effective_batch_mean=float(np.mean(sizes)) if sizes else 0.,
                supported=sum(x[0] for x in output), wrong=sum(x[1] for x in output)), output


def run(old, snapshot, reader, transfer_results, frozen, output):
    if (output.exists() or indexed.digest(transfer_results) != COLD_RESULTS or indexed.digest(frozen) != transfer.REFERENCE
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("Require frozen inputs and a new output from Slurm CUDA")
    previous = json.loads(transfer_results.read_text())
    freeze = json.loads(frozen.read_text())
    rows, _, _ = transfer.cohort(old, snapshot)
    if (previous["doi_inventory"] != [row[0] for row in rows]
            or previous["transfer_capture_sha256"] != indexed.digest(snapshot / "capture.json")):
        raise ValueError("Changed transfer corpus")
    gate = transfer.load_gate(freeze["routes"]["character"]["selected"]["gate"], cascade.Gate)
    if gate is None or gate.features != (0,):
        raise ValueError("Expected the frozen character-only gate")
    catalog = {}
    for doi, text, *_ in rows:
        clauses = indexed.compile_note(text)
        grams = [indexed.char_grams(c.action) for c in clauses]
        catalog[doi] = dict(clauses=clauses, grams=[(g, indexed.gram_norm(g)) for g in grams])
    tokenizer, model, reader_info = indexed.load_reader(reader)
    if reader_info["revision"] != freeze["reader_revision"]:
        raise ValueError("Changed reader revision")
    labels = [tokenizer.encode(word, add_special_tokens=False) for word in ("1", "0")]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError("Expected two distinct outcome tokens")
    engines = model, tokenizer, [ids[0] for ids in labels]
    cutoff = freeze["reader_cutoff"]
    for n in (1, 4, 8):
        logits(rows[:n], engines)
    result = {}
    for rate in RATES:
        result[rate] = []
        for i in range(TRIALS):
            order, offsets = trace(rows, rate, 20261119 + i)
            arms = list(ARMS)
            arms = arms[i:] + arms[:i]
            outcomes, details = {}, {}
            for arm in arms:
                char_route, size = ARMS[arm]
                details[arm], outcomes[arm] = trial(rows, catalog, engines, gate if char_route else None,
                                                     cutoff, order, offsets, size)
            for arm in ARMS:
                char_route, size = ARMS[arm]
                stem = "character" if char_route else "reader"
                expected = previous["test"][stem]
                same = sum(x == y for x, y in zip(outcomes[arm], outcomes["pair_" + stem], strict=True))
                details[arm]["pair_parity"] = same
                if size == 1 and (details[arm]["supported"], details[arm]["wrong"]) != (
                        2 * expected["accepted_supported"], 2 * expected["accepted_wrong"]):
                    raise ValueError("Pair-model outcomes changed from independent replay")
            result[rate].append(details)
            print("Completed", rate, i, flush=True)
    summary = {rate: {arm: dict(
        latency_p50_p95_p99_ms=np.median([trial[arm]["latency_p50_p95_p99_ms"] for trial in rounds], axis=0).tolist(),
        **{k: float(np.median([trial[arm].get(k, 0) for trial in rounds])) for k in
           ("regular_gpu_batches", "reader_pairs", "singleton_rechecks", "waiting_p95_ms",
            "source_to_completion_rps", "scheduled_rps", "producer_lag_p95_ms", "pair_parity",
            "effective_batch_mean", "supported", "wrong")})
        for arm in ARMS} for rate, rounds in result.items()}
    indexed.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                    script_sha256=indexed.digest(Path(__file__)), transfer_sha256=COLD_RESULTS,
                                    freeze_sha256=transfer.REFERENCE, reader_revision=reader_info["revision"],
                                    papers=len(rows), requests_per_trial=2 * len(rows), repeats=TRIALS,
                                    batch_wait_ms=1000 * WAIT, recheck_band=RECHECK_BAND,
                                    summary=summary, trials=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "snapshot", "reader", "transfer_results", "frozen", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
