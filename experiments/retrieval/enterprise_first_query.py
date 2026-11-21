"""Time live development searches with interchangeable requirement readers."""

import argparse
from collections import Counter
import json
from itertools import permutations
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_guarded_parsing import GuardedReader
from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans

ARMS = ("independent", "cache", "margin_recheck")
CUTOFF = 2.0
K = 5
SOURCE_DEPTH = 2
SOURCE_CHARS = 800
CRITERION_CHARS = 320
VERIFY_BATCH = 16
TASK = (
    "Determine whether the document explicitly describes prior work supporting the technical capability "
    "or experience in the query. A related topic or promise of future work is insufficient."
)
EMBED_TASK = "Retrieve records of prior expert work relevant to the enterprise technical request."


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def score_steps(steps, reranker):
    try:
        pairs, task = next(steps)
        while True:
            pairs, task = steps.send(reranker.score_pairs(pairs, task=task))
    except StopIteration as completed:
        return completed.value


def verify_candidates(hits, sources, criteria, reranker):
    return score_steps(verification_steps(hits, sources, criteria), reranker)


def verification_steps(hits, sources, criteria):
    pairs = []
    for hit in hits:
        for source in hit["evidence"][:SOURCE_DEPTH]:
            row = sources[source["id"]]
            if row["expert_id"] != hit["id"]:
                raise AssertionError("Source belongs to a different expert")
            excerpt = row["text"][:SOURCE_CHARS]
            if excerpt:
                citation = (source["id"], 0, len(excerpt))
                pairs.extend((hit["id"], i, query, excerpt, citation) for i, query in enumerate(criteria)
                             if len(query) <= CRITERION_CHARS)
    witness = {}
    clipped = withheld = 0
    for offset in range(0, len(pairs), VERIFY_BATCH):
        batch = pairs[offset:offset + VERIFY_BATCH]
        margins, truncated = yield [(row[2], row[3]) for row in batch], TASK
        clipped += truncated
        if truncated:
            withheld += len(batch)
            continue
        for (expert, criterion, _, _, citation), margin in zip(batch, margins, strict=True):
            key = (expert, criterion)
            if not np.isfinite(margin):
                raise ValueError("Nonfinite support margin")
            if margin > 0 and (key not in witness or margin > witness[key][0]):
                witness[key] = (margin, citation)
    decisions = []
    for hit in hits:
        expert = hit["id"]
        citations = tuple(witness.get((expert, i), (None, None))[1] for i in range(len(criteria)))
        decisions.append((expert, all(citations), citations))
    decisions.sort(key=lambda row: not row[1])
    return decisions, len(pairs), clipped, withheld


def evaluate(text, method, reader, embedding, index, sources, reranker):
    return score_steps(evaluation_steps(text, method, reader, embedding, index, sources), reranker)


def evaluation_steps(text, method, reader, embedding, index, sources):
    start = perf_counter()
    clauses = spans(text)
    parsed = reader.evaluate(text, clauses, method, cutoff=CUTOFF)
    qualifications = tuple((clause["start"], clause["end"]) for clause, label
                           in zip(clauses, parsed["labels"], strict=True)
                           if ROLES[int(label)] == "prior_qualification")
    criteria = (text.strip().splitlines()[0][:CRITERION_CHARS],) + tuple(text[a:b] for a, b in qualifications)
    parsed_at = perf_counter()
    vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
    embedded_at = perf_counter()
    hits = index.search_vector(vector, k=K)["hits"]
    retrieved_at = perf_counter()
    decisions, pairs, clipped, withheld = yield from verification_steps(hits, sources, criteria)
    finished = perf_counter()
    return dict(labels=parsed["labels"], qualifications=qualifications,
                retrieved=tuple(hit["id"] for hit in hits), decisions=decisions,
                pairs=pairs, clipped_pairs=clipped, withheld_pairs=withheld, rechecked=parsed["rechecked"],
                supported=sum(row[1] for row in decisions),
                overlong_qualifications=sum(len(criterion) > CRITERION_CHARS for criterion in criteria),
                times=dict(parse=parsed_at - start, query_embedding=embedded_at - parsed_at,
                           retrieve=retrieved_at - embedded_at, verify_and_rank=finished - retrieved_at,
                           total=finished - start))


def summarize(results, arms=ARMS):
    report = {}
    rng = np.random.default_rng(719)
    sample = rng.integers(len(results), size=(4999, len(results)))
    paired_reference = np.array([row["independent"]["times"]["total"] for row in results])
    for arm in arms:
        counts = Counter()
        for row in results:
            reference, current = row["independent"], row[arm]
            counts["clause_label_changes"] += int((reference["labels"] != current["labels"]).sum())
            counts["qualification_set_changes"] += reference["qualifications"] != current["qualifications"]
            counts["retrieved_order_changes"] += reference["retrieved"] != current["retrieved"]
            decisions, baseline = current["decisions"], reference["decisions"]
            counts["rank_changes"] += [r[0] for r in decisions] != [r[0] for r in baseline]
            counts["supported_set_changes"] += {r[0] for r in decisions if r[1]} != {r[0] for r in baseline if r[1]}
            counts["citation_or_unresolved_changes"] += (
                reference["qualifications"] != current["qualifications"] or decisions != baseline)
            counts["queries_with_provisional_support"] += current["supported"] > 0
            for key in ("pairs", "clipped_pairs", "withheld_pairs", "supported", "overlong_qualifications", "rechecked"):
                counts[key] += current[key]
        quantiles = {name: (1000 * np.quantile([row[arm]["times"][name] for row in results], [0.5, 0.95])).tolist()
                     for name in results[0][arm]["times"]}
        total = np.array([row[arm]["times"]["total"] for row in results])
        ratio = np.quantile(paired_reference[sample], 0.95, axis=1) / np.quantile(total[sample], 0.95, axis=1)
        report[arm] = dict(counts, p50_p95_ms=quantiles,
                           p95_speedup_ci95=np.quantile(ratio, [0.025, 0.975]).tolist())
    return report


def run(queries, corpus, models, output, qualification_guard=False):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output path and Slurm CUDA allocation")
    requests = read_rows(queries)
    if len(requests) != 80 or len({row["id"] for row in requests}) != 80 or any(
            row["split"] != "development" for row in requests):
        raise ValueError("Expected 80 unique development requests")
    records = read_rows(corpus)
    sources = {row["id"]: row for row in records}
    if len(sources) != len(records) or any(row["split"] != "development" for row in records):
        raise ValueError("Expected unique development sources")
    torch.backends.cuda.matmul.allow_tf32 = False
    embedding_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(embedding_path)
    embedding = QwenTorchEmbedding(embedding_path)
    start = perf_counter()
    vectors = embedding.encode([row["text"] for row in records])
    reference = ExactOwnedVectorIndex(vectors, list(sources), [[row["expert_id"]] for row in records])
    index = GpuOwnedVectorIndex(reference)
    index_seconds = perf_counter() - start
    print(f"Indexed {len(sources)} sources in {index_seconds:.1f} s", flush=True)
    reader = GuardedReader(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    warmup = "Find an expert in optical inspection. Prior experience testing polymer films is required."
    arms = (*ARMS, "qualification_recheck") if qualification_guard else ARMS
    for method in arms:
        evaluate(warmup, method, reader, embedding, index, sources, verifier)
    results = []
    if qualification_guard:
        # Balance arm positions and first-order carryover on the 80 requests.
        orders = tuple(tuple(arms[(i + offset) % len(arms)] for i in (0, 1, 3, 2))
                       for offset in range(len(arms)))
    else:
        orders = tuple(permutations(arms))
    for i, request in enumerate(requests):
        results.append({method: evaluate(request["text"], method, reader, embedding, index, sources, verifier)
                        for method in orders[i % len(orders)]})
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} requests across all reader arms", flush=True)
    model_manifests = {
        "qwen3-embedding-0.6b": embedding_path / "checkpoint-provenance.json",
        "qwen3-4b-instruct-2507": models / "qwen3-4b-instruct-2507" / "local-provenance.json",
        "qwen3-reranker-0.6b": models / "qwen3-reranker-0.6b" / "checkpoint-provenance.json",
    }
    write_json(output, dict(requests=len(requests), sources=len(sources), experts=len(reference.owners),
                            development_sha256=digest(queries), corpus_sha256=digest(corpus),
                            model_manifests={name: digest(path) for name, path in model_manifests.items()},
                            embedding_revision=provenance["revision"], cutoff=CUTOFF, k=K,
                            source_depth=SOURCE_DEPTH, source_chars=SOURCE_CHARS, criterion_chars=CRITERION_CHARS,
                            index_seconds=index_seconds, hardware=torch.cuda.get_device_name(),
                            torch_version=str(torch.__version__), slurm_job_id=os.environ["SLURM_JOB_ID"],
                            arms=summarize(results, arms)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("queries", "corpus", "models", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--qualification-guard", action="store_true",
                        help="Also recheck cached blocks near the qualification boundary.")
    run(**vars(parser.parse_args()))
