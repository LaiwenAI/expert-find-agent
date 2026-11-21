"""Replay the held role prompt comparison through a single-GPU HTTP service."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import Future
from copy import copy
import json
import os
from pathlib import Path
from queue import Queue
from random import Random
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.distilled_role_recheck import route
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import (
    TOPIC_TASK, Catalog, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_one_role_recheck import RoleReader, replace
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.joint_loopback_http import Handler, run_clients
from experiments.credit.joint_request_queue import Request
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_loopback_http import Server


ARMS = ("static", "full", "name_only")
RATES = (35, 65, 80)
REPEATS = 2


def workload(catalog, rows, wording):
    phrases = phrase_bank(wording)
    lookup = {}
    for row in rows:
        excerpts = [catalog.articles[paper]["clue"] for paper in row["papers"]]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for clues in (excerpts, [catalog.keywords(clue) for clue in excerpts]):
            request = Request("elife", json.dumps({"requirements": [
                {"topic": clue, "role": role} for clue, role in zip(clues, text, strict=True)
            ]}))
            if request in lookup and any(row[key] != lookup[request][key] for key in ("papers", "roles", "gold")):
                raise ValueError("Identical requests have conflicting references")
            lookup[request] = row
    return lookup


def arrivals_for(requests, rate, repeats=REPEATS):
    random = Random(20261121)
    trace = []
    for _ in range(repeats):
        queries = list(requests)
        random.shuffle(queries)
        trace.extend(queries)
    arrivals = [0.]
    for _ in trace[1:]:
        arrivals.append(arrivals[-1] + random.expovariate(rate))
    return trace, arrivals


def search(request, catalog, encoder, banks, reader, bands, cutoff):
    items = json.loads(request.payload)["requirements"]
    clues = [item["topic"] for item in items]
    tasks = [item["role"] for item in items]
    prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
    prompts += [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in tasks]
    vectors = encoder.encode(prompts)
    topic = catalog.rank(vectors[:2], clues, None)
    pair, _ = select(catalog, topic, {}, [set(), set()], "anchored", None)
    owners, roles, calls, tokens = [], [], 0, 0
    if pair is not None:
        scope = {"papers": pair, "roles": (None, None)}
        scores = {name: vectors[2:] @ banks[name].T for name in bands}
        if reader is None:
            sets = required_roles(scope, scores, "prototype_band", bands)
        else:
            sets, _, slot, approved = route(catalog.articles, scope, scores, bands)
            if approved:
                choice, margin, tokens = reader.read(tasks[slot], sets[slot])
                calls = 1
                if choice in approved and margin > cutoff:
                    sets = replace(sets, slot, choice)
        owners = candidates(catalog, pair, sets)
        roles = [sorted(values) for values in sets]
    return json.dumps(dict(papers=pair, owners=owners, roles=roles, calls=calls, tokens=tokens))


def worker(pending, execute, warm, ready):
    try:
        for request in warm:
            execute(request)
        ready.set_result(True)
    except Exception as exc:
        ready.set_exception(exc)
        return
    while (item := pending.get()) is not None:
        request, arrived, future = item
        started = perf_counter()
        try:
            content = execute(request)
            future.set_result((content, 1, (started - arrived) * 1000, (perf_counter() - started) * 1000))
        except Exception as exc:
            future.set_exception(exc)


def replay(execute, trace, arrivals, warm):
    with Server(("127.0.0.1", 0), Handler) as server:
        server.pending = Queue()
        server.panels = {"elife"}
        ready = Future()
        gpu = Thread(target=worker, args=(server.pending, execute, warm, ready))
        gpu.start()
        try:
            ready.result(timeout=60)
            network = Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
            network.start()
            try:
                return asyncio.run(run_clients(server.server_port, trace, arrivals))
            finally:
                server.shutdown()
                network.join()
        finally:
            server.pending.put(None)
            gpu.join()


def summarize(trace, arrivals, rows, elapsed, labels, repeats):
    outcomes, stable = Counter(), {}
    complete = [row for row in rows if "response" in row]
    for request, record in zip(trace, rows, strict=True):
        if "response" not in record:
            continue
        answer = json.loads(record["response"])
        truth = labels[request]
        options = answer["roles"]
        correct_roles = len(options) == 2 and all(role in choices for role, choices in zip(truth["roles"], options))
        parties = set(answer["owners"])
        expected = set(truth["gold"])
        source = answer["papers"] == list(truth["papers"])
        exact = bool(expected) and source and parties == expected and correct_roles
        adverse = bool(parties) and (not source or not correct_roles or bool(parties - expected))
        proof = (answer["papers"], answer["owners"], answer["roles"])
        if request in stable and stable[request] != proof:
            raise ValueError("Repeated request produced different evidence")
        stable[request] = proof
        outcomes.update(exact=exact, adverse=adverse, issued=bool(parties),
                        positives=bool(expected), calls=answer["calls"], tokens=answer["tokens"])
    elapsed_before_drain = arrivals[-1]
    return dict(
        counts=dict(outcomes), requests=len(trace), expected_repeats=repeats,
        offered_rps=(len(trace) - 1) / elapsed_before_drain, achieved_rps=len(complete) / elapsed,
        failures=dict(Counter(row["failure"] for row in rows if "failure" in row)),
        makespan_seconds=elapsed, drain_seconds=max(0., elapsed - elapsed_before_drain),
        p95_ms={key: float(np.quantile([row[key] for row in complete], .95)) for key in
                ("total_ms", "client_queue_ms", "server_queue_ms", "compute_ms")} if complete else {},
    ), stable


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, cases, panel = build(args.projection, json.loads(args.manifest.read_bytes()))
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    names = copy(full)
    names.definitions = ROLES
    readers = {"static": None, "full": full, "name_only": names}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    catalog = Catalog(articles, encoder)
    labels = workload(catalog, cases, wording)
    results, observed = {}, {}
    arms = ARMS[::-1] if args.reverse else ARMS
    for index, rate in enumerate(RATES):
        results[str(rate)] = {}
        trace, arrivals = arrivals_for(labels, rate)
        # Rotate arm order and use an identical trace at each offered rate.
        for arm in arms[index:] + arms[:index]:
            def execute(request):
                return search(request, catalog, encoder, banks, readers[arm], bands, cutoff)
            rows, elapsed = replay(execute, trace, arrivals, trace[:24])
            detail, responses = summarize(trace, arrivals, rows, elapsed, labels, REPEATS)
            if arm in observed:
                prior = observed[arm]
                detail["changed_vs_first_rate"] = sum(prior[req] != answer for req, answer in responses.items() if req in prior)
            else:
                observed[arm] = responses
            results[str(rate)][arm] = detail
            print(rate, arm, detail["p95_ms"], flush=True)
    paired = observed["full"].keys() & observed["name_only"].keys()
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), manifest_sha256=digest(args.manifest), projection_sha256=digest(args.projection),
        panel=panel, distinct_requests=len(labels), embedding_revision=checkpoint["revision"],
        reader_revision=full.provenance["revision"], gpu=torch.cuda.get_device_name(), reverse=args.reverse,
        outputs_compared=len(paired), changed_full_to_names=sum(observed["full"][r] != observed["name_only"][r] for r in paired),
        results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "manifest", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
