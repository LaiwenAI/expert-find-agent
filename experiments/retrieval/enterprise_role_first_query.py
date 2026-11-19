"""Measure live first-query recovery through a development-only role index."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_project_role_rescue as role
from experiments.retrieval import enterprise_role_pool_expansion as expanded
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("whole", "deeper_on_abstain", "global_on_abstain", "global_always")
ROLE_READS = 2


def compile_roles(dataset, corpus, output):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    experts, originals = (role.profile_subset(dataset, key, manifest) for key in ("experts", "corpus"))
    projected = review.index(rows(corpus), lambda row: row["id"])
    if originals.keys() != projected.keys() or any(
        row["split"] != "development" or originals[key]["text"] != row["text"]
        or originals[key]["expert_id"] != row["expert_id"] for key, row in projected.items()
    ):
        raise ValueError("The source projection must match the development inventory")
    windows, _ = expanded.build_index(experts, originals)
    write(output, dict(
        scope="development", corpus_sha256=digest(corpus),
        compiler_sha256=digest(Path(expanded.__file__)),
        windows=[{key: window[key] for key in ("expert", "view")} for window in windows],
    ))


def build_vectors(corpus, embedding, vectors, metadata):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_gap_verification import SOURCE_INVENTORY

    require_cuda(vectors, metadata)
    if digest(corpus) != SOURCE_INVENTORY:
        raise ValueError("Changed development inventory")
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    started = perf_counter()
    encoded = encoder.encode([row["text"] for row in rows(corpus)])
    with vectors.open("xb") as output:
        vectors.chmod(0o600)
        np.save(output, encoded, allow_pickle=False)
    write(metadata, dict(
        corpus_sha256=SOURCE_INVENTORY, checkpoint=checkpoint["revision"], vectors_sha256=digest(vectors),
        rows=len(encoded), seconds=perf_counter() - started,
    ))


def require_cuda(*outputs):
    import torch

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or any(path.exists() for path in outputs):
        raise ValueError("Require Slurm CUDA and new output paths")


def load_role_index(path, corpus_digest, sources):
    raw = json.loads(path.read_bytes())
    if (raw["scope"], raw["corpus_sha256"], raw["compiler_sha256"]) != (
        "development", corpus_digest, digest(Path(expanded.__file__))
    ):
        raise ValueError("Changed role-window inventory")
    windows = raw["windows"]
    postings = defaultdict(list)
    for i, item in enumerate(windows):
        view = item["view"]
        original = sources[view["id"]]
        if (original["expert_id"] != item["expert"] or not view["roles"]
                or original["text"][view["start"]:view["end"]] != view["text"]):
            raise ValueError("Role window does not match its original source")
        item["terms"] = role.terms(view["text"])
        for term in sorted(item["terms"]):
            postings[term].append(i)
    return windows, postings


def recover(criteria, ranked, issued, role_index, verifier, arm, task, maximum):
    if arm == "whole" or (issued and arm != "global_always") or not maximum:
        return [], 0, False, 0
    windows, postings = role_index
    shortlist = {eid: rank for rank, eid in enumerate(ranked)}
    hits = {i for term in role.terms(criteria[0]) for i in postings.get(term, ())}
    eligible = (windows[i] for i in sorted(hits) if windows[i]["expert"] not in issued
                and (arm != "deeper_on_abstain" or windows[i]["expert"] in shortlist))
    options = expanded.select(eligible, criteria, shortlist)[:ROLE_READS]
    if not options:
        return [], 0, True, 0
    inputs = [(criterion, item["view"]["text"]) for item in options for criterion in criteria]
    margins, clipped = verifier.score_pairs(inputs, task=task)
    if not np.isfinite(margins).all():
        raise ValueError("Nonfinite rescue margin")
    if clipped:
        return [], len(inputs), True, clipped
    result = {}
    for i, item in enumerate(options):
        low, high = i * len(criteria), (i + 1) * len(criteria)
        if all(score > 0 for score in margins[low:high]):
            view = item["view"]
            result.setdefault(item["expert"], [view["id"], view["start"], view["end"]])
    added = [dict(expert=eid, citations=[citation]) for eid, citation in result.items()]
    return added[:maximum], len(inputs), True, 0


def run(queries, corpus, roles, vectors, vector_meta, models, output, private):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import (
        CUTOFF, CRITERION_CHARS, EMBED_TASK, K, ROLES, TASK, spans, verify_candidates,
    )
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    require_cuda(output, private)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed pinned development projection")
    torch.backends.cuda.matmul.allow_tf32 = False
    requests = rows(queries)
    sources = review.index(rows(corpus), lambda row: row["id"])
    windows = load_role_index(roles, SOURCE_INVENTORY, sources)
    cached = json.loads(vector_meta.read_bytes())
    encoder_path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(encoder_path)
    if (cached["corpus_sha256"], cached["checkpoint"], cached["vectors_sha256"], cached["rows"]) != (
        SOURCE_INVENTORY, checkpoint["revision"], digest(vectors), len(sources)
    ):
        raise ValueError("Source vectors do not match the corpus or encoder")
    encoder = QwenTorchEmbedding(encoder_path)
    matrix = np.load(vectors, allow_pickle=False)
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        matrix, list(sources), [[row["expert_id"]] for row in sources.values()],
    ))
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    reader = Reranker(models / "qwen3-reranker-0.6b")

    def one(text, arm):
        started = perf_counter()
        clauses = spans(text)
        labels = parser.evaluate(text, clauses, "broadcast_prefix", CUTOFF)["labels"]
        qualification = [(item["start"], item["end"]) for item, label in zip(clauses, labels, strict=True)
                         if ROLES[int(label)] == "prior_qualification"]
        criteria = [text.strip().splitlines()[0][:CRITERION_CHARS], *(text[a:b] for a, b in qualification)]
        vector = encoder.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
        hits = index.search_vector(vector, k=K)["hits"]
        decisions, checks, clipped, withheld = verify_candidates(hits, sources, criteria, reader)
        initial = [dict(expert=eid, citations=[list(c) for c in dict.fromkeys(citations)])
                   for eid, success, citations in decisions if success]
        ranked = [eid for eid, _, _ in decisions]
        extra, pairs, opened, truncated = recover(
            criteria, ranked, {item["expert"] for item in initial}, windows, reader, arm, TASK, K - len(initial),
        ) if all(len(c) <= CRITERION_CHARS for c in criteria) else ([], 0, False, 0)
        return dict(
            initial=initial, ranked=ranked, extra=extra, qualifications=qualification,
            primary_pairs=checks, primary_clipped=clipped, primary_withheld=withheld,
            rescue_pairs=pairs, rescue_clipped=truncated, invoked=opened,
            ms=(perf_counter() - started) * 1000,
        )

    warmup = "Find an expert in optical inspection with prior polymer film testing work."
    for arm in ARMS:
        one(warmup, arm)
    results = []
    for i, query in enumerate(requests):
        shift = i % len(ARMS)
        arms = {arm: one(query["text"], arm) for arm in ARMS[shift:] + ARMS[:shift]}
        for state in arms.values():
            if any(state[key] != arms["whole"][key] for key in ("initial", "ranked", "qualifications")):
                raise ValueError("Primary search diverged between policies")
        results.append(dict(id=query["id"], arms=arms))
        if (i + 1) % 20 == 0:
            print(f"Measured {i+1} development queries", flush=True)
    aggregate = describe(results)
    fingerprint = dict(
        source_sha256=SOURCE_INVENTORY, queries_sha256=DEVELOPMENT, role_index_sha256=digest(roles),
        vectors_sha256=cached["vectors_sha256"], code_sha256=digest(Path(__file__)),
    )
    write(private, dict(results=results, **fingerprint))
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), **fingerprint,
        models=dict(encoder=checkpoint["revision"], parser=parser.readers["bf16"].provenance["revision"],
                    verifier=reader.provenance["revision"]),
        requests=len(requests), indexed_sources=len(sources), indexed_role_windows=len(windows[0]),
        index_build_seconds=cached["seconds"], role_read_cap=ROLE_READS, metrics=aggregate,
    ))


def describe(results):
    metrics = {}
    first = np.array([row["arms"]["whole"]["ms"] for row in results])
    rng = np.random.default_rng(1719)
    paired = rng.integers(len(first), size=(6000, len(first)))
    for arm in ARMS:
        values = [row["arms"][arm] for row in results]
        latencies = np.array([value["ms"] for value in values])
        diffs = np.quantile(latencies[paired], .95, axis=1) - np.quantile(first[paired], .95, axis=1)
        metrics[arm] = dict(
            issued=sum(bool(v["initial"] or v["extra"]) for v in values),
            initially_unanswered_now_issued=sum(not v["initial"] and bool(v["extra"]) for v in values),
            added_slots=sum(len(v["extra"]) for v in values),
            added_outside_initial_rank=sum(sum(e["expert"] not in v["ranked"] for e in v["extra"]) for v in values),
            **{key: sum(v[key] for v in values) for key in (
                "primary_pairs", "primary_clipped", "primary_withheld", "rescue_pairs", "rescue_clipped", "invoked",
            )},
            p50_p95_ms=np.quantile(latencies, [.5, .95]).tolist(),
            paired_p95_delta_interval_ms=np.quantile(diffs, [.025, .975]).tolist(),
        )
    return metrics


def packet(dataset, trace, output):
    if output.exists():
        raise FileExistsError(output)
    observed = json.loads(trace.read_bytes())
    if observed["code_sha256"] != digest(Path(__file__)):
        raise ValueError("Changed execution trace")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    queries, experts, corpus = (role.profile_subset(dataset, key, manifest) for key in ("queries", "experts", "corpus"))
    methods = {arm: {qid: [] for qid in queries} for arm in ARMS}
    qualifiers = {}
    if {record["id"] for record in observed["results"]} != set(queries):
        raise ValueError("Unexpected development query inventory")
    for record in observed["results"]:
        qid = record["id"]
        raw = queries[qid]["text"]
        qualifiers[qid] = [raw[a:b] for a, b in record["arms"]["whole"]["qualifications"]]
        for arm, responses in record["arms"].items():
            for item in responses["initial"] + responses["extra"]:
                citations = [dict(source_id=s, start=a, end=b) for s, a, b in item["citations"]]
                methods[arm][qid].append((item["expert"], review.sources(corpus, item["expert"], citations)))
    forms, mapping, predictions, audit = review.packet_rows(queries, experts, methods)
    for form, mapping_row in zip(forms, mapping, strict=True):
        form["prior_qualifications"] = qualifiers[mapping_row["query_id"]]
    output.mkdir(mode=0o700)
    blind = output / "blind-review.jsonl"
    exact = output / "private-mapping.jsonl"
    write(blind, forms, lines=True)
    write(exact, mapping, lines=True)
    files = [blind, exact]
    for arm, values in predictions.items():
        path = output / f"{arm}-predictions.jsonl"
        write(path, values, lines=True)
        files.append(path)
    write(output / "manifest.json", dict(queries=len(queries), candidate_views=len(forms),
                                        outputs_sha256={f.name: digest(f) for f in files}))
    validation = review.check(output, blind)
    if any(set(counts) != {"missing"} for counts in validation["counts"].values()):
        raise ValueError("Expected a blank blinded review")
    for reviewer in ("a", "b"):
        write(output / f"reviewer-{reviewer}.jsonl", forms, lines=True)
    print(json.dumps(dict(review_views=len(forms), human_labels=0, **audit)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subcommands = cli.add_subparsers(dest="command", required=True)
    commands = {
        "roles": (compile_roles, ("dataset", "corpus", "output")),
        "vectors": (build_vectors, ("corpus", "embedding", "vectors", "metadata")),
        "run": (run, ("queries", "corpus", "roles", "vectors", "vector_meta", "models", "output", "private")),
        "packet": (packet, ("dataset", "trace", "output")),
    }
    for command, (_, args) in commands.items():
        child = subcommands.add_parser(command)
        for arg in args:
            child.add_argument(arg, type=Path)
    args = vars(cli.parse_args())
    commands[args.pop("command")][0](**args)
