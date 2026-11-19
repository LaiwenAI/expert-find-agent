"""Compare contiguous criterion windows in the live enterprise development search."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from experiments.retrieval import enterprise_shared_qualification as route
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("reference", "lead", "criterion")
WINDOW_CHARS = 320
BOUNDARY = re.compile(r"[。！？；;!?\n]+|[.]\s+")


def select(text, criterion, arm):
    if arm == "reference" or len(text) <= WINDOW_CHARS:
        return 0, len(text)
    if arm == "lead":
        return 0, WINDOW_CHARS
    terms = set(tokens(criterion))
    boundaries = [match.end() for match in BOUNDARY.finditer(text)]
    last = len(text) - WINDOW_CHARS
    starts = sorted({0, last, *range(0, last, WINDOW_CHARS // 2), *(i for i in boundaries if i <= last)})
    start = max(starts, key=lambda pos: len(terms.intersection(tokens(text[pos:pos + WINDOW_CHARS]))))
    return start, start + WINDOW_CHARS


def primary(hits, sources, criteria, reader, arm):
    if arm == "reference":
        return route.base.verify_candidates(hits, sources, criteria, reader)
    pairs = []
    for pair in route.plan(hits, sources, criteria):
        start, end = select(pair.evidence, pair.text, arm)
        selected = pair.evidence[start:end]
        pairs.append(replace(pair, evidence=selected, citation=(pair.citation[0], start, end)))
    witness = {}
    clipped = withheld = 0
    for offset in range(0, len(pairs), route.base.VERIFY_BATCH):
        batch = pairs[offset:offset + route.base.VERIFY_BATCH]
        margins, truncated = reader.score_pairs([pair.model_key for pair in batch], task=route.base.TASK)
        if not np.isfinite(margins).all():
            raise ValueError("Invalid verifier margin")
        clipped += truncated
        if truncated:
            withheld += len(batch)
            continue
        for pair, margin in zip(batch, margins, strict=True):
            key = pair.expert, pair.criterion
            if margin > 0 and (key not in witness or margin > witness[key][0]):
                witness[key] = margin, pair.citation
    output = []
    for hit in hits:
        expert = hit["id"]
        citations = tuple(witness.get((expert, i), (None, None))[1] for i in range(len(criteria)))
        output.append((expert, all(citations), citations))
    output.sort(key=lambda row: not row[1])
    return output, len(pairs), clipped, withheld


def search(text, arm, parser, encoder, index, sources, role_index, reader):
    from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans

    start = perf_counter()
    clauses = spans(text)
    parsed = parser.evaluate(text, clauses, "broadcast_prefix", cutoff=route.base.CUTOFF)
    qualifications = tuple((c["start"], c["end"]) for c, label in zip(clauses, parsed["labels"], strict=True)
                           if ROLES[int(label)] == "prior_qualification")
    criteria = (text.strip().splitlines()[0][:route.base.CRITERION_CHARS],) + tuple(text[a:b] for a,b in qualifications)
    vector = encoder.encode([f"Instruct: {route.base.EMBED_TASK}\nQuery: {text}"])[0]
    hits = index.search_vector(vector, k=route.base.K)["hits"]
    reader.change("guarded_rescue")
    verify_at = perf_counter()
    decisions, pairs, clipped, withheld = primary(hits, sources, criteria, reader, arm)
    primary_ms = (perf_counter() - verify_at) * 1000
    primary_counts = dict(reader.counts)
    issued = [(eid, citations) for eid, accepted, citations in decisions if accepted]
    ranked = [eid for eid, _, _ in decisions]
    reader.counts.clear()
    rescue, _, invoked, rescue_clipped = route.role.recover(
        criteria, ranked, {eid for eid, _ in issued}, role_index, reader, "global_on_abstain", route.base.TASK,
        route.base.K - len(issued)) if all(len(c) <= route.base.CRITERION_CHARS for c in criteria) else ([], 0, False, 0)
    candidates = [eid for eid, _ in issued] + [item["expert"] for item in rescue]
    citations = tuple((eid, tuple(c[0] for c in claims)) for eid, claims in issued)
    return dict(retrieved=[hit["id"] for hit in hits], qualifications=qualifications, issued=candidates,
                primary_citations=citations, primary_spans=issued, rescue=rescue, invoked=invoked, pairs=pairs,
                clipped=clipped + rescue_clipped, withheld=withheld, primary_counts=primary_counts,
                rescue_counts=dict(reader.counts), primary_ms=primary_ms, ms=(perf_counter() - start) * 1000)


def summarize(rows):
    report = {}
    anchor = [row["reference"] for row in rows]
    for arm in ARMS:
        decisions = [row[arm] for row in rows]
        compare = list(zip(anchor, decisions, strict=True))
        if any(a["pairs"] != b["pairs"] or a["retrieved"] != b["retrieved"] or
               a["qualifications"] != b["qualifications"] for a, b in compare):
            raise ValueError("The primary pool or workload changed")
        report[arm] = dict(
            queries=len(decisions), issued_queries=sum(bool(r["issued"]) for r in decisions),
            changed_people_or_order=sum(a["issued"] != b["issued"] for a,b in compare),
            gained_queries=sum(not a["issued"] and bool(b["issued"]) for a,b in compare),
            lost_queries=sum(bool(a["issued"]) and not b["issued"] for a,b in compare),
            gained_people=sum(len(set(b["issued"]) - set(a["issued"])) for a,b in compare),
            lost_people=sum(len(set(a["issued"]) - set(b["issued"])) for a,b in compare),
            changed_primary_sources=sum(a["primary_citations"] != b["primary_citations"] for a,b in compare),
            changed_primary_spans=sum(a["primary_spans"] != b["primary_spans"] for a,b in compare),
            rescue_changed=sum(a["rescue"] != b["rescue"] for a,b in compare),
            rescue_invoked=sum(b["invoked"] for b in decisions),
            clipped=sum(b["clipped"] for b in decisions), withheld=sum(b["withheld"] for b in decisions),
            primary=dict(sum((Counter(b["primary_counts"]) for b in decisions), Counter())),
            rescue=dict(sum((Counter(b["rescue_counts"]) for b in decisions), Counter())),
            primary_p95_ms=float(np.quantile([b["primary_ms"] for b in decisions], .95)),
            live_p95_ms=float(np.quantile([b["ms"] for b in decisions], .95)))
    return report


def run(queries, corpus, roles, vectors, meta, models, output):
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    route.role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Only the development inputs are eligible")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {doc["id"]: doc for doc in rows(corpus)}
    enc = models / "qwen3-embedding-0.6b"
    pinned = verify_checkpoint(enc)
    stored = json.loads(meta.read_bytes())
    if (SOURCE_INVENTORY, pinned["revision"], digest(vectors), len(sources)) != tuple(
            stored[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector index provenance differs")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    canonical = Reranker(models / "qwen3-reranker-0.6b")
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    fixture = (parser, QwenTorchEmbedding(enc), index, sources,
               route.role.load_role_index(roles, SOURCE_INVENTORY, sources), route.EncodedReader(canonical))
    for arm in ARMS:
        search("Find an expert in sensor calibration.", arm, *fixture)
    observations = []
    for i, request in enumerate(rows(queries)):
        offset = i % len(ARMS)
        observations.append({arm: search(request["text"], arm, *fixture) for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1} private development requests", flush=True)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), models=dict(encoder=pinned["revision"],
            reranker=canonical.provenance["revision"], parser=parser.readers["bf16"].provenance["revision"]),
        char_budget=WINDOW_CHARS, metrics=summarize(observations)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
