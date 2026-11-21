"""Measure one-wave and two-wave evidence checks through loopback HTTP."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import Future
from hashlib import sha256
from http.server import BaseHTTPRequestHandler
from itertools import product
import json
import os
from pathlib import Path
from queue import Queue
from random import Random
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.community import joint_answer_witness as base
from experiments.community.expected_support_dispatch import best_seed, next_source
from experiments.credit.joint_loopback_http import run_clients, Server
from experiments.credit.joint_request_queue import Request
from experiments.credit.live_joint_source_request import simple_order, verify_embedding
from experiments.credit.vector_hybrid_rrf import RankFusion
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.retrieval.lrbench_frozen_baselines import build_lexical

ARMS = ("staged", "static_split", "static_batched", "staged_sparse", "static_sparse")
RATES = (5, 14)
REQUESTS = 54
CLIENT_DEADLINE_SECONDS = 8


class Search:
    def __init__(self, documents, encoder, reader, scratch):
        self.ids = sorted(documents, key=int)
        self.documents = documents
        self.owners = {doc: row["owner"] for doc, row in documents.items()}
        self.encoder = encoder
        self.reader = reader
        texts = [documents[doc]["answer"] for doc in self.ids]
        lexical = {}
        for position, text in enumerate(texts):
            while text in lexical:
                text += "\n"
            lexical[text] = position
        self.fusion = RankFusion(range(len(texts)), build_lexical(lexical, scratch))
        self.index = encoder.encode(texts)

    def ranking(self, texts):
        scores = self.encoder.encode(texts, query=True) @ self.index.T
        return tuple({self.ids[doc]: i for i, doc in enumerate(self.fusion.rank(score, text, "array_part")[:base.DEPTH], 1)}
                     for text, score in zip(texts, scores, strict=True))

    def verify(self, texts, inputs, margins):
        if not inputs:
            return 0
        scores, _, tokens = self.reader.score_pairs(
            [(texts[side], self.documents[doc]["answer"][:base.WINDOW_CHARS]) for side, doc in inputs],
            task=base.TASK, count_tokens=True,
        )
        if len(scores) != len(inputs) or not np.isfinite(scores).all():
            raise ValueError("Invalid verifier outputs")
        margins.update(zip(inputs, map(float, scores), strict=True))
        return tokens

    def search(self, texts, arm):
        ranks = self.ranking(texts)
        anchors, extras = simple_order(ranks, depth=base.DEPTH)
        pair = (0, 1)
        margins = {}
        tokens, waves = 0, 0

        def check(inputs):
            nonlocal tokens, waves
            if inputs:
                tokens += self.verify(texts, inputs, margins)
                waves += 1

        def all_checks(sources):
            return [(side, doc) for doc, side in product(sources, range(2))]

        def proof_checks(proposals):
            needed = {(side, doc) for proposal in proposals for side, doc in enumerate(proposal)}
            return [(side, doc) for side, doc in all_checks(selected)
                    if (side, doc) in needed and (side, doc) not in margins]

        if arm.startswith("staged"):
            first = all_checks(anchors)
            if arm == "staged_sparse":
                first = [(side, doc) for side, doc in first if doc in ranks[side]]
            check(first)
            extra = base.stage_extra(extras, anchors, pair, ranks, self.owners, margins)
            selected = [*anchors, extra]
            candidates = base.paired_sources(ranks, selected, self.owners)
            check(proof_checks(candidates) if arm == "staged_sparse" else all_checks([extra]))
        else:
            weights = tuple({doc: 1 / rank for doc, rank in order.items()} for order in ranks)
            seed = best_seed(ranks, self.owners, anchors)
            extra = next_source(seed, sorted(set().union(*ranks)), weights, self.owners, ranks)
            selected = [*seed, extra]
            candidates = base.paired_sources(ranks, selected, self.owners)
            if arm == "static_split":
                check(all_checks(seed))
                check(all_checks([extra]))
            elif arm == "static_sparse":
                check(proof_checks(candidates))
            else:
                check(all_checks(selected))
        limit = len(margins)
        if len(set(selected)) != 3 or limit > 6 or (not arm.endswith("_sparse") and limit != 6):
            raise ValueError("Exceeded service budget")
        proof, confidence = base.choose(pair, ranks, candidates, margins)
        return json.dumps(dict(
            sources=selected, proof=proof, confidence=confidence, tokens=tokens, checks=limit, waves=waves,
            read_excerpts=len({doc for _, doc in margins}),
        )).encode()



class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path != "/search":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 32768:
                raise ValueError("Unexpected request size")
            texts = json.loads(self.rfile.read(length))["questions"]
            if not isinstance(texts, list) or len(texts) != 2 or not all(
                isinstance(text, str) and text.strip() for text in texts
            ):
                raise ValueError("Expected two questions")
        except (ValueError, TypeError, KeyError):
            self.send_error(400)
            return
        future = Future()
        self.server.pending.put((texts, perf_counter(), future))
        try:
            payload, queue_ms, compute_ms = future.result(timeout=CLIENT_DEADLINE_SECONDS)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Server-Timing", f"1,{queue_ms:.6f},{compute_ms:.6f},0")
            self.end_headers()
            self.wfile.write(payload)
        except TimeoutError:
            self.send_error(503)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.send_error(500)


def replay(search, arm, trace, arrivals):
    with Server(("127.0.0.1", 0), Handler) as server:
        server.pending = Queue()
        ready = Future()

        def worker():
            try:
                for request in trace[:4]:
                    search.search(json.loads(request.payload)["questions"], arm)
            except Exception as error:
                ready.set_exception(error)
                return
            ready.set_result(None)
            while (task := server.pending.get()) is not None:
                texts, queued, future = task
                started = perf_counter()
                try:
                    answer = search.search(texts, arm)
                    future.set_result((answer, (started - queued) * 1000, (perf_counter() - started) * 1000))
                except Exception as error:
                    future.set_exception(error)

        compute = Thread(target=worker)
        compute.start()
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
            compute.join()


def workload(rows, cases, seed):
    subset = sorted(cases, key=lambda pair: sha256(f"community-http:{pair[0]}:{pair[1]}".encode()).digest())[:REQUESTS]
    requests = []
    expected = {}
    for pair in subset:
        payload = json.dumps({"questions": [rows[q]["text"] for q in pair]})
        request = Request("mathoverflow", payload)
        known = [rows[q]["known"] for q in pair]
        if request in expected:
            raise ValueError("Duplicated request")
        requests.append(request)
        expected[request] = known
    random = Random(seed)
    random.shuffle(requests)
    gaps = [random.expovariate(1) for _ in requests[1:]]
    return requests, expected, np.r_[0., np.cumsum(gaps)]


def summarize(results, trace, expected, arrivals):
    complete = {}
    summaries = {}
    for arm, (rows, elapsed) in results.items():
        failures = Counter(row["failure"] for row in rows if "failure" in row)
        if not any("response" in row for row in rows):
            raise ValueError(f"No completed responses for {arm}: {failures}")
        decoded = {request: json.loads(row["response"]) for request, row in zip(trace, rows, strict=True)
                   if "response" in row}
        complete[arm] = decoded
        successful = [row for row in rows if "response" in row]
        proofs = {req: answer for req, answer in decoded.items() if answer["proof"] is not None}

        def known(req):
            return all(doc in labels for doc, labels in zip(proofs[req]["proof"], expected[req], strict=True))

        summaries[arm] = dict(
            completed=len(decoded), failures=dict(failures), achieved_rps=len(decoded) / elapsed,
            p95_ms={key: float(np.quantile([row[key] for row in successful], .95))
                    for key in ("total_ms", "network_ms", "client_queue_ms", "server_queue_ms", "compute_ms")},
            verified_pairs=sum(answer["checks"] for answer in decoded.values()),
            source_slots=3 * len(decoded), read_excerpts=sum(answer["read_excerpts"] for answer in decoded.values()),
            verifier_waves=sum(answer["waves"] for answer in decoded.values()),
            verifier_tokens=sum(answer["tokens"] for answer in decoded.values()),
            issued=len(proofs), known=sum(map(known, proofs)),
            proof_signature=sha256(json.dumps([
                (decoded[req]["sources"], decoded[req]["proof"]) if req in decoded else None for req in trace
            ]).encode()).hexdigest(),
        )
    split, batched = (complete[arm] for arm in ("static_split", "static_batched"))
    shared = set(split) & batched.keys()
    if any(split[req]["sources"] != batched[req]["sources"] or split[req]["tokens"] != batched[req]["tokens"]
           for req in shared):
        raise ValueError("Identical static plan changed evidence or prompt tokens across schedules")
    comparisons = {}
    for baseline, treatment in (
        ("static_split", "static_batched"), ("staged", "static_batched"),
        ("static_batched", "static_sparse"), ("staged", "staged_sparse"), ("staged_sparse", "static_sparse"),
    ):
        first, second = complete[baseline], complete[treatment]
        common = first.keys() & second.keys()
        selected = [{req: route[req] for req in trace if req in common and route[req]["proof"] is not None}
                    for route in (first, second)]
        quota = min(map(len, selected))
        matched_known = [
            sum(all(doc in labels for doc, labels in zip(output[req]["proof"], expected[req], strict=True))
                for req in sorted(output, key=lambda req: -output[req]["confidence"])[:quota])
            for output in selected
        ]
        comparisons[f"{treatment}_vs_{baseline}"] = dict(
            shared=len(common), matched_quota=quota, matched_known=matched_known,
            proof_changes=sum(first[req]["proof"] != second[req]["proof"] for req in common),
            plan_changes=sum(first[req]["sources"] != second[req]["sources"] for req in common),
        )
    return dict(requests=len(trace), offered_rps=(len(trace) - 1) / arrivals[-1],
                results=summaries, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "embedding", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--offset", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a Slurm GPU and fresh output")
    raw, docs, snapshot = base.load_snapshot(args.snapshot)
    frozen = json.loads(args.reference.read_bytes())
    if snapshot != frozen["panels"]["mathoverflow_original"]["snapshot_sha256"]:
        raise ValueError("Changed MathOverflow snapshot")
    rows, cases, _, _ = base.freeze_cases(raw, docs, True)
    revision = verify_embedding(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    if revision != frozen["embedding_checkpoint"] or reader.provenance["revision"] != frozen["reranker_checkpoint"]:
        raise ValueError("Changed checkpoint")
    trace, expected, arrivals = workload(rows, cases, 20261121 + args.offset)
    results = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        search = Search(docs, encoder, reader, Path(scratch))
        rates = RATES[::-1] if args.offset else RATES
        for i, rate in enumerate(rates):
            measured = {}
            offset = (args.offset + i) % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            for arm in order:
                measured[arm] = replay(search, arm, trace, arrivals / rate)
                print(f"Finished {rate} arrivals/s: {arm}", flush=True)
            results[str(rate)] = summarize(measured, trace, expected, arrivals / rate)
    write_json(args.output, dict(
        traces=results, snapshot_sha256=snapshot, script_sha256=digest(Path(__file__)),
        reference_sha256=digest(args.reference), embedding=revision, reranker=reader.provenance["revision"],
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, offset=args.offset, job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    main()
