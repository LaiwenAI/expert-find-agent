"""Replay guarded verifier packing across enterprise requests on one GPU."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from queue import Empty
from time import perf_counter

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval.enterprise_cross_precision_load import WARMUP, trace, wire_result, workload
from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, K, TASK, evaluation_steps
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import recovery_steps, require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write


def steps(text, parser, encoder, index, sources, roles):
    primary = yield from evaluation_steps(text, "broadcast_prefix", parser, encoder, index, sources)
    initial = {eid for eid, accepted, _ in primary["decisions"] if accepted}
    criteria = [text.strip().splitlines()[0][:CRITERION_CHARS],
                *(text[a:b] for a, b in primary["qualifications"])]
    extra, _, invoked, clipped = [], 0, False, 0
    if all(len(c) <= CRITERION_CHARS for c in criteria):
        extra, _, invoked, clipped = yield from recovery_steps(
            criteria, [row[0] for row in primary["decisions"]], initial, roles, "global_on_abstain",
            TASK, K - len(initial),
        )
    return dict(qualifications=primary["qualifications"], retrieved=primary["retrieved"],
                decisions=dict(initial=[
                    dict(expert=eid, citations=[list(c) for c in dict.fromkeys(citations)])
                    for eid, accepted, citations in primary["decisions"] if accepted
                ], primary_detail=primary["decisions"], fallback=extra, invoked=invoked,
                    clipped=primary["clipped_pairs"] + clipped))


def score_wave(reader, requests, work):
    responses, ragged = {}, []
    with reader.torch.inference_mode():
        for request, (pairs, task) in requests.items():
            sequences, truncated = reader.encode(pairs, task)
            lengths = [len(row["input_ids"]) for row in sequences]
            padded, actual = len(lengths) * max(lengths), sum(lengths)
            work.update(logical_batches=1, logical_pairs=len(pairs), logical_tokens=actual)
            if len(lengths) > 1 and actual < packed.TOKEN_RATIO * padded:
                ragged.append((request, sequences, truncated, padded))
            else:
                responses[request] = reader.padded(sequences, padded), truncated
        if not ragged:
            return responses
        all_rows = [seq for _, sequences, _, _ in ragged for seq in sequences]
        margins, tokens = reader.forward(reader.backbone, reader.weight, all_rows)
        work.update(packed_forwards=1, packed_request_batches=len(ragged), packed_tokens=tokens,
                    shared_forwards=int(len(ragged) > 1))
        offset = 0
        for request, sequences, truncated, padded in ragged:
            end = offset + len(sequences)
            subset = margins[offset:end]
            offset = end
            if any(abs(margin) <= packed.SIGN_MARGIN for margin in subset):
                work.update(guard_rechecks=1)
                subset = reader.padded(sequences, padded)
            responses[request] = subset, truncated
    return responses


def run_group(texts, inputs, reader, work):
    searches = {i: steps(text, *inputs) for i, text in enumerate(texts)}
    values = dict.fromkeys(searches)
    while searches:
        wave = {}
        for i, search in searches.items():
            try:
                wave[i] = search.send(values[i])
            except StopIteration as completed:
                yield i, wire_result(completed.value)
        searches = {i: searches[i] for i in wave}
        if wave:
            values = score_wave(reader, wave, work)


def replay(texts, fixture, indices, times, capacity):
    *inputs, reader = fixture
    work = Counter()
    def serve(pending, search, ready):
        try:
            list(run_group([WARMUP], inputs, reader, work))
            work.clear()
            reader.counts.clear()
            ready.set_result(True)
        except Exception as exc:
            ready.set_exception(exc)
            return
        while (first := pending.get()) is not None:
            group = [first]
            for _ in range(capacity - 1):
                try:
                    item = pending.get_nowait()
                except Empty:
                    break
                if item is None:
                    pending.put(None)
                    break
                group.append(item)
            started = perf_counter()
            work[f"request_group_{len(group)}"] += 1
            try:
                for i, body in run_group([item[0] for item in group], inputs, reader, work):
                    _, arrived, future = group[i]
                    future.set_result((body, (started-arrived)*1000, (perf_counter()-started)*1000))
            except Exception as exc:
                for _, _, future in group:
                    if not future.done():
                        future.set_exception(exc)
    bodies, timing = workload(texts, indices, times, None, worker_fn=serve)
    timing["work"] = dict(work)
    timing["work"]["padded_forwards"] = reader.counts["padded_batches"]
    timing["work"]["padded_tokens"] = reader.counts["executed_tokens"]
    timing["issued"] = sum(bool(row["initial"] or row["fallback"]) for body in bodies
                           for row in [json.loads(body)["decisions"]])
    return bodies, timing


def run(queries, corpus, roles, vectors, meta, models, serial, output, seed, reverse):
    require_cuda(output)
    texts, fixture, fingerprint = init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    reference_inputs = {i: wire_result(response) for i, text in enumerate(texts)
                        for result in [packed.one(text, "guarded_rescue", *fixture)]
                        for response in [dict(
                            qualifications=result["qualifications"], retrieved=result["ranked"],
                            decisions=dict(initial=result["initial"], primary_detail=result["decisions"],
                                           fallback=result["extra"], invoked=result["invoked"],
                                           clipped=result["clipped"]),
                        )]}
    results = []
    for offset, rate in enumerate((2.2, 3.2)):
        indices, times = trace(len(texts), rate, seed)
        order = (1, 4) if bool(offset) == reverse else (4, 1)
        runs = {}
        for capacity in order:
            replies, timing = replay(texts, fixture, indices, times, capacity)
            timing["changed_responses"] = sum(reply != reference_inputs[index]
                                              for index, reply in zip(indices, replies, strict=True))
            runs[capacity] = timing
            print(rate, capacity, timing["changed_responses"], timing["quantiles_ms"]["total_ms"][1], flush=True)
        if runs[1]["changed_responses"]:
            raise ValueError("Serial staged execution must match the original guarded implementation")
        results.append(dict(nominal_rate=rate, execution_order=order, runs=runs))
    write(output, dict(slurm_job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
                       seed=seed, changed_responses_from_affinity=None, results=results, **fingerprint))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--seed", required=True, type=int)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
