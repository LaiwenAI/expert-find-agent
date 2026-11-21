"""Benchmark exact lexical owner reduction on the private enterprise development split."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_lexical_pool import OwnedLexicalPool
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import enterprise_hybrid_rescue as prior
from experiments.retrieval.enterprise_first_query import (
    CRITERION_CHARS, CUTOFF, EMBED_TASK, K, ROLES, SOURCE_DEPTH, spans, verify_candidates,
)
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.sepef_first_query import lexical_index

ARMS = ("group_all_sources", "defer_source_sort")


def stage_summary(before, after):
    keys = sorted(before[0])
    if set(keys) != set(after[0]) or len(before) != len(after):
        raise ValueError("Timing rows must align")
    choice = np.random.default_rng(20261120).integers(len(before), size=(4000, len(before)))
    output = {}
    for key in keys:
        baseline = np.asarray([entry[key] for entry in before])
        candidate = np.asarray([entry[key] for entry in after])
        delta = np.quantile(candidate[choice], .95, axis=1) - np.quantile(baseline[choice], .95, axis=1)
        output[key] = dict(
            legacy_p50_p95_ms=np.quantile(baseline, [.5, .95]).tolist(),
            deferred_p50_p95_ms=np.quantile(candidate, [.5, .95]).tolist(),
            paired_p95_delta_ci95_ms=np.quantile(delta, [.025, .975]).tolist(),
        )
    return output


def score_query(text, ranking, parser, encoder, verifier, sources):
    begin = perf_counter()
    clauses = spans(text)
    parsed = parser.evaluate(text, clauses, "broadcast_prefix", CUTOFF)
    qualifications = [(clause["start"], clause["end"]) for clause, label in zip(clauses, parsed["labels"], strict=True)
                      if ROLES[int(label)] == "prior_qualification"]
    criteria = [text.strip().splitlines()[0][:CRITERION_CHARS], *(text[a:b] for a, b in qualifications)]
    after_parser = perf_counter()
    vector = encoder.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
    after_encoding = perf_counter()
    hits = ranking(text, vector)
    after_rank = perf_counter()
    decisions, pairs, clipped, withheld = verify_candidates(hits, sources, criteria, verifier)
    finished = perf_counter()
    response = dict(qualifications=qualifications, hits=hits, decisions=decisions, verification_pairs=pairs,
                    clipped=clipped, withheld=withheld)
    times = dict(total=(finished-begin)*1000, parser=(after_parser-begin)*1000,
                 query_encoding=(after_encoding-after_parser)*1000, retrieval=(after_rank-after_encoding)*1000,
                 verification=(finished-after_rank)*1000)
    return response, times


def fast_rank(text, vector, index, pool):
    hits = index.search_vector(vector, k=prior.DEPTH)["hits"]
    return pool.hybrid(text, hits, k=K, source_depth=SOURCE_DEPTH, fusion_depth=prior.DEPTH)


def run(args):
    prior.role.require_cuda(args.output)
    ref = json.loads(args.reference.read_bytes())
    if (digest(args.queries), digest(args.corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Require private development projections")
    if digest(args.vectors) != ref["vectors_sha256"]:
        raise ValueError("Changed source vectors")
    requests = rows(args.queries)
    if len(requests) != 80 or len({q["id"] for q in requests}) != 80 or any(
            q["split"] != "development" for q in requests):
        raise ValueError("Require exactly 80 distinct development requests")
    sources = prior.review.index(rows(args.corpus), lambda row: row["id"])
    ids = list(sources)
    owners = [source["expert_id"] for source in sources.values()]
    embedding_path = args.models / "qwen3-embedding-0.6b"
    revision = verify_checkpoint(embedding_path)["revision"]
    meta = json.loads(args.vector_meta.read_bytes())
    if (meta["corpus_sha256"], meta["checkpoint"], meta["vectors_sha256"], meta["rows"]) != (
            SOURCE_INVENTORY, revision, ref["vectors_sha256"], len(sources)):
        raise ValueError("Source vector provenance differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(np.load(args.vectors, allow_pickle=False), ids,
                                                     [[owner] for owner in owners]))
    encoder = QwenTorchEmbedding(embedding_path)
    parser = ComparisonParser(args.models / "qwen3-4b-instruct-2507")
    verifier = Reranker(args.models / "qwen3-reranker-0.6b")
    models = dict(embedding=revision, parser=parser.readers["bf16"].provenance["revision"],
                  verifier=verifier.provenance["revision"])
    if models != ref["models"]:
        raise ValueError("Model revisions differ from the archived pipeline")
    with TemporaryDirectory(dir=args.scratch, prefix="owned-lexical-") as folder:
        clock = perf_counter()
        lexical = lexical_index([row["text"] for row in sources.values()], Path(folder))
        original_build = perf_counter() - clock
        clock = perf_counter()
        backed = OwnedLexicalPool(lexical, owners, ids)
        additional_build = perf_counter() - clock
        rankers = {
            "group_all_sources": lambda text, vector: prior.hybrid_hits(text, vector, index, lexical, owners, ids, K, SOURCE_DEPTH),
            "defer_source_sort": lambda text, vector: fast_rank(text, vector, index, backed),
        }
        warmup = "Find an expert in optical inspection with prior polymer film testing work."
        for arm in ARMS:
            score_query(warmup, rankers[arm], parser, encoder, verifier, sources)
        order = list(reversed(requests)) if args.reverse else requests
        execution = {arm: {} for arm in ARMS}
        component = {arm: {} for arm in ARMS}
        receipts = []
        for i, request in enumerate(order):
            text, qid = request["text"], request["id"]
            use = ARMS if (i+args.reverse) % 2 == 0 else ARMS[::-1]
            response = {}
            for arm in use:
                response[arm], execution[arm][qid] = score_query(text, rankers[arm], parser, encoder, verifier, sources)
            if response[ARMS[0]] != response[ARMS[1]]:
                raise ValueError(f"Exact hybrid outputs changed on development request {i}")
            receipts.append((qid, response[ARMS[0]]))
            if (i+1) % 20 == 0:
                print(f"Matched {i+1} development responses", flush=True)
        for i, request in enumerate(order):
            text, qid = request["text"], request["id"]
            query = encoder.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
            times = {arm: [] for arm in ARMS}
            for run in range(6):
                use = ARMS if (i+run+args.reverse) % 2 == 0 else ARMS[::-1]
                comparison = {}
                for arm in use:
                    start = perf_counter()
                    comparison[arm] = rankers[arm](text, query)
                    times[arm].append((perf_counter()-start)*1000)
                if comparison[ARMS[0]] != comparison[ARMS[1]]:
                    raise ValueError("Cached-vector retrieval scores or ordering changed")
            for arm in ARMS:
                component[arm][qid] = dict(retrieval=float(np.median(times[arm])))
        write(args.output, dict(
            requests=len(requests), sources=len(sources), owner_count=len(set(owners)),
            proof_receipt_sha256=sha256(json.dumps(sorted(receipts), sort_keys=True).encode()).hexdigest(),
            comparisons=dict(exact_candidate_and_proof_matches=len(receipts),
                             requests_issued=sum(any(d[1] for d in row["decisions"]) for _, row in receipts),
                             supported_slots=sum(sum(d[1] for d in row["decisions"]) for _, row in receipts),
                             verification_pairs=sum(row["verification_pairs"] for _, row in receipts)),
            complete=stage_summary(*[[execution[arm][qid] for qid in sorted(execution[arm])] for arm in ARMS]),
            component=stage_summary(*[[component[arm][qid] for qid in sorted(component[arm])] for arm in ARMS]),
            original_lexical_build_seconds=original_build, original_lexical_payload_bytes=lexical.array_bytes,
            deferred_index_build_seconds=additional_build, deferred_index_payload_bytes=backed.indexing_bytes,
            model_revisions=models, corpus_sha256=digest(args.corpus), queries_sha256=digest(args.queries),
            vectors_sha256=digest(args.vectors), reference_sha256=digest(args.reference),
            script_sha256=digest(Path(__file__)), aggregation_sha256=digest(
                Path(__file__).parents[2]/"src"/"expert_search"/"owned_lexical_pool.py"),
            job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reverse=args.reverse,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "vectors", "vector_meta", "models", "scratch", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
