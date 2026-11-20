"""Compare enterprise role rescue with a common full-corpus lexical and dense baseline."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.channel_fusion import fuse
from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_project_role_rescue as profile
from experiments.retrieval import enterprise_role_first_query as role
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("dense", "hybrid", "deeper_on_abstain", "global_on_abstain")
DEPTH = 50


def lexical_hits(lexical, text, owners, source_ids):
    grouped = defaultdict(list)
    for doc, score in lexical.document_scores(text).items():
        idx = int(doc)
        grouped[owners[idx]].append((score, source_ids[idx]))
    for evidence in grouped.values():
        evidence.sort(key=lambda row: (-row[0], row[1]))
    return sorted(grouped, key=lambda oid: (-grouped[oid][0][0], oid))[:DEPTH], grouped


def hybrid_hits(text, vector, index, lexical, owners, source_ids, k, source_depth):
    dense_hits = index.search_vector(vector, k=DEPTH)["hits"]
    dense = {hit["id"]: hit["evidence"] for hit in dense_hits}
    sparse_order, sparse = lexical_hits(lexical, text, owners, source_ids)
    experts = fuse({"dense": list(dense), "lexical": sparse_order}, k=k, depth=DEPTH)
    result = []
    for hit in experts:
        expert = hit["id"]
        dense_ids = [row["id"] for row in dense.get(expert, ())]
        sparse_ids = [source_id for _, source_id in sparse.get(expert, ())[:source_depth]]
        sources = fuse({"dense": dense_ids, "lexical": sparse_ids}, k=source_depth, depth=source_depth)
        result.append(dict(id=expert, evidence=sources))
    return result


def summary(observations):
    all_rows = {arm: [item["arms"][arm] for item in observations] for arm in ARMS}
    base = all_rows["hybrid"]
    draws = np.random.default_rng(20261120).integers(len(base), size=(4000, len(base)))
    base_p95 = np.quantile(np.asarray([row["ms"] for row in base])[draws], .95, axis=1)
    output = {}
    for arm, values in all_rows.items():
        times = np.asarray([row["ms"] for row in values])
        diffs = np.quantile(times[draws], .95, axis=1) - base_p95
        output[arm] = dict(
            queries=len(values), issued=sum(bool(row["initial"] or row["extra"]) for row in values),
            slots=sum(len(row["initial"])+len(row["extra"]) for row in values),
            added=sum(len(row["extra"]) for row in values),
            outside_initial=sum(sum(item["expert"] not in row["ranked"] for item in row["extra"]) for row in values),
            rescue_pairs=sum(row["rescue_pairs"] for row in values),
            primary_pairs=sum(row["primary_pairs"] for row in values),
            changed_ranked=sum(a["ranked"] != b["ranked"] for a,b in zip(values, base, strict=True)),
            p95_ms=float(np.quantile(times, .95)),
            paired_p95_vs_hybrid_ci95_ms=np.quantile(diffs, [.025,.975]).tolist(),
        )
    global_runs, deeper = all_rows["global_on_abstain"], all_rows["deeper_on_abstain"]
    matched = [(a,b) for a,b in zip(global_runs,deeper,strict=True) if a["rescue_pairs"] == b["rescue_pairs"] > 0]
    return dict(arms=output, matched_nonzero_budget=dict(
        requests=len(matched), pairs=sum(a["rescue_pairs"] for a,_ in matched),
        global_issued=sum(bool(a["extra"]) for a,_ in matched),
        deeper_issued=sum(bool(b["extra"]) for _,b in matched),
    ))


def run(queries, corpus, role_windows, vectors, vector_meta, models, scratch, output, trace):
    import torch

    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import (
        CRITERION_CHARS, CUTOFF, EMBED_TASK, K, ROLES, SOURCE_DEPTH, TASK, spans, verify_candidates,
    )
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
    from experiments.retrieval.sepef_first_query import lexical_index

    role.require_cuda(output, trace)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Require the unchanged development projections")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = review.index(rows(corpus), lambda row: row["id"])
    ids = list(sources)
    owners = [row["expert_id"] for row in sources.values()]
    embedding_path = models / "qwen3-embedding-0.6b"
    revision = verify_checkpoint(embedding_path)["revision"]
    cache = json.loads(vector_meta.read_bytes())
    if tuple(cache[k] for k in ("corpus_sha256","checkpoint","vectors_sha256","rows")) != (
            SOURCE_INVENTORY, revision, digest(vectors), len(sources)):
        raise ValueError("Cached embeddings do not match the development inventory")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(np.load(vectors, allow_pickle=False), ids,
                                                     [[owner] for owner in owners]))
    window_index = role.load_role_index(role_windows, SOURCE_INVENTORY, sources)
    embedding = QwenTorchEmbedding(embedding_path)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")

    with TemporaryDirectory(prefix="enterprise-hybrid-", dir=scratch) as directory:
        started = perf_counter()
        lexical = lexical_index([row["text"] for row in sources.values()], Path(directory))
        build_seconds = perf_counter() - started

        def one(text, arm):
            begin = perf_counter()
            clauses = spans(text)
            parsed = parser.evaluate(text, clauses, "broadcast_prefix", CUTOFF)
            qualification = [(clause["start"], clause["end"]) for clause, label in zip(
                clauses, parsed["labels"], strict=True) if ROLES[int(label)] == "prior_qualification"]
            criteria = [text.strip().splitlines()[0][:CRITERION_CHARS], *(text[a:b] for a,b in qualification)]
            vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
            hits = (index.search_vector(vector, k=K)["hits"] if arm == "dense"
                    else hybrid_hits(text, vector, index, lexical, owners, ids, K, SOURCE_DEPTH))
            decisions, pairs, clipped, withheld = verify_candidates(hits, sources, criteria, verifier)
            initial = [dict(expert=eid, citations=[list(c) for c in dict.fromkeys(citations)])
                       for eid, accepted, citations in decisions if accepted]
            ranked = [item[0] for item in decisions]
            extra, rescue_pairs, invoked, rescue_clipped = ([], 0, False, 0)
            if arm in ("deeper_on_abstain", "global_on_abstain") and all(
                    len(criterion) <= CRITERION_CHARS for criterion in criteria):
                extra, rescue_pairs, invoked, rescue_clipped = role.recover(
                    criteria, ranked, {item["expert"] for item in initial}, window_index,
                    verifier, arm, TASK, K - len(initial),
                )
            return dict(initial=initial, extra=extra, ranked=ranked, qualifications=qualification,
                        primary_pairs=pairs, primary_clipped=clipped, primary_withheld=withheld,
                        rescue_pairs=rescue_pairs, rescue_clipped=rescue_clipped, invoked=invoked,
                        ms=(perf_counter()-begin)*1000)

        for arm in ARMS:
            one("Find an expert in optical inspection with prior polymer film testing work.", arm)
        observations = []
        for count, request in enumerate(rows(queries)):
            shift = count % len(ARMS)
            compared = {arm: one(request["text"], arm) for arm in ARMS[shift:] + ARMS[:shift]}
            for arm, data in compared.items():
                if data["qualifications"] != compared["hybrid"]["qualifications"]:
                    raise ValueError("Parsing changed across retrieval methods")
                if arm != "dense" and any(data[key] != compared["hybrid"][key] for key in ("ranked","initial")):
                    raise ValueError("Hybrid primary search changed across rescue arms")
            observations.append(dict(id=request["id"], arms=compared))
            if (count+1) % 20 == 0:
                print(f"Measured {count+1} development queries", flush=True)
    metadata = dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                    code_sha256=digest(Path(__file__)), queries_sha256=DEVELOPMENT,
                    corpus_sha256=SOURCE_INVENTORY, vectors_sha256=cache["vectors_sha256"],
                    role_index_sha256=digest(role_windows), lexical_build_seconds=build_seconds,
                    lexical_postings_bytes=lexical.array_bytes, sources=len(sources), k=K,
                    source_depth=SOURCE_DEPTH, fusion_depth=DEPTH, rescue_cap=role.ROLE_READS,
                    models=dict(embedding=revision, parser=parser.readers["bf16"].provenance["revision"],
                                verifier=verifier.provenance["revision"]))
    write(trace, dict(metadata=metadata, observations=observations))
    write(output, dict(metadata, **summary(observations)))


def packet(dataset, trace, output):
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY

    private = json.loads(trace.read_bytes())
    metadata = private["metadata"]
    if (metadata["code_sha256"], metadata["queries_sha256"], metadata["corpus_sha256"]) != (
            digest(Path(__file__)), DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Trace provenance differs")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    queries, experts, corpus = (profile.profile_subset(dataset, key, manifest)
                                for key in ("queries","experts","corpus"))
    observed = review.index(private["observations"], lambda row: row["id"])
    if observed.keys() != queries.keys():
        raise ValueError("Unexpected development query inventory")
    methods = {arm: {} for arm in ARMS}
    for qid, record in observed.items():
        if set(record["arms"]) != set(ARMS):
            raise ValueError("Incomplete comparison")
        for arm, result in record["arms"].items():
            methods[arm][qid] = [(item["expert"],review.sources(corpus, item["expert"],
                [dict(source_id=source, start=begin, end=end) for source,begin,end in item["citations"]]))
                for item in result["initial"] + result["extra"]]
    forms, mapping, predictions, counts = review.packet_rows(queries, experts, methods)
    for form, row in zip(forms, mapping, strict=True):
        qid = row["query_id"]
        form["prior_qualifications"] = [queries[qid]["text"][a:b]
            for a,b in observed[qid]["arms"]["hybrid"]["qualifications"]]
    output.mkdir(mode=0o700)
    blind = output / "blind-review.jsonl"
    mapping_path = output / "private-mapping.jsonl"
    write(blind, forms, lines=True)
    write(mapping_path, mapping, lines=True)
    files = [blind, mapping_path]
    for arm, values in predictions.items():
        path = output / f"{arm}-predictions.jsonl"
        write(path, values, lines=True)
        files.append(path)
    write(output / "manifest.json", dict(queries=len(queries), candidate_views=len(forms),
                    outputs_sha256={path.name: digest(path) for path in files}))
    validation = review.check(output, blind)
    if any(set(values) != {"missing"} for values in validation["counts"].values()):
        raise ValueError("Expected blank review fields")
    for reviewer in ("a", "b"):
        write(output / f"reviewer-{reviewer}.jsonl", forms, lines=True)
    print(json.dumps(dict(views=len(forms), human_labels=0, **counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for handler, names in ((run, ("queries","corpus","role_windows","vectors","vector_meta","models","scratch","output","trace")),
                           (packet, ("dataset","trace","output"))):
        sub = commands.add_parser(handler.__name__)
        for name in names:
            sub.add_argument(name,type=Path)
    args = vars(parser.parse_args())
    handler = {"run":run,"packet":packet}[args.pop("command")]
    handler(**args)
