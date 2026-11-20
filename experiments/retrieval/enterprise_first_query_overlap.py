"""Overlap independent demand parsing and full-request retrieval on one GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import enterprise_shared_qualification as search
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("serial", "parallel")


class FirstQuery:
    def __init__(self, parser, encoder, index, sources, role_index, verifier, pool):
        self.parser, self.encoder, self.index = parser, encoder, index
        self.sources, self.role_index, self.verifier, self.pool = sources, role_index, verifier, pool
        self.stream = torch.cuda.Stream()
        torch.cuda.synchronize()

    def retrieve(self, text, asynchronous=False):
        started = perf_counter()
        stream = self.stream if asynchronous else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            vector = self.encoder.encode([f"Instruct: {search.base.EMBED_TASK}\nQuery: {text}"])[0]
            hits = self.index.search_vector(vector, k=search.base.K)["hits"]
        return hits, started, perf_counter()

    def run(self, text, arm):
        started = perf_counter()
        pending = self.pool.submit(self.retrieve, text, True) if arm == "parallel" else None
        parsed_at = perf_counter()
        try:
            clauses = spans(text)
            parsed = self.parser.evaluate(text, clauses, "broadcast_prefix", cutoff=search.base.CUTOFF)
            qualifications = tuple((c["start"], c["end"]) for c, label in zip(clauses, parsed["labels"], strict=True)
                                   if ROLES[int(label)] == "prior_qualification")
            criteria = (text.strip().splitlines()[0][:search.base.CRITERION_CHARS],) + tuple(
                text[a:b] for a, b in qualifications)
            parsed_end = perf_counter()
        finally:
            if pending is not None:
                hits, retrieval_at, retrieval_end = pending.result()
        if pending is None:
            hits, retrieval_at, retrieval_end = self.retrieve(text)
        joined = perf_counter()
        self.verifier.change("guarded_rescue")
        decisions, metadata = search.execute("automatic", hits, self.sources, criteria, self.verifier)
        initial = [dict(expert=eid, citations=[list(span) for span in dict.fromkeys(citations)])
                   for eid, accepted, citations in decisions if accepted]
        ranked = [eid for eid, _, _ in decisions]
        self.verifier.counts.clear()
        rescue, _, invoked, clipped = search.role.recover(
            criteria, ranked, {item["expert"] for item in initial}, self.role_index, self.verifier,
            "global_on_abstain", search.base.TASK, search.base.K - len(initial)
        ) if all(len(c) <= search.base.CRITERION_CHARS for c in criteria) else ([], 0, False, 0)
        end = perf_counter()
        overlap = max(0., min(parsed_end, retrieval_end) - max(parsed_at, retrieval_at))
        return dict(
            labels=tuple(map(int, parsed["labels"])), qualifications=qualifications, hits=hits,
            decisions=decisions, initial=initial, rescue=rescue, invoked=invoked,
            qualification_route=metadata, clipped=clipped,
            timing=dict(parser=(parsed_end - parsed_at) * 1000, retrieval=(retrieval_end - retrieval_at) * 1000,
                        join=(joined - started) * 1000, total=(end - started) * 1000,
                        overlap=overlap * 1000))


def summarize(observations):
    ref, parallel = ([row[arm] for row in observations] for arm in ARMS)
    rng = np.random.default_rng(20261125)
    bootstrap = rng.integers(len(ref), size=(6000, len(ref)))
    times = [np.asarray([r["timing"]["total"] for r in arm]) for arm in (ref, parallel)]
    delta = np.quantile(times[1][bootstrap], .95, axis=1) - np.quantile(times[0][bootstrap], .95, axis=1)
    return dict(
        metrics={name: dict(
            queries=len(arm),
            issued=sum(bool(row["initial"] or row["rescue"]) for row in arm),
            p50_p95_ms={key: np.quantile([r["timing"][key] for r in arm], [.5, .95]).tolist()
                        for key in arm[0]["timing"]})
                 for name, arm in zip(ARMS, (ref, parallel), strict=True)},
        paired_total_p95_delta_ci95=np.quantile(delta, [.025, .975]).tolist(),
        changed_requests={key: sum(a[key] != b[key] for a, b in zip(ref, parallel, strict=True))
                          for key in ("labels", "qualifications", "hits", "decisions", "initial", "rescue", "invoked",
                                      "qualification_route", "clipped")})


def run(queries, corpus, roles, vectors, metadata, models, output):
    search.role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Unexpected development data")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {doc["id"]: doc for doc in rows(corpus)}
    meta = json.loads(metadata.read_bytes())
    enc = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(enc)
    if (SOURCE_INVENTORY, provenance["revision"], digest(vectors), len(sources)) != tuple(
            meta[key] for key in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector cache provenance differs")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    verifier = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    role_index = search.role.load_role_index(roles, SOURCE_INVENTORY, sources)
    observations = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        query = FirstQuery(parser, QwenTorchEmbedding(enc), index, sources, role_index, verifier, pool)
        for arm in ARMS:
            query.run("Find an expert in optical sensors. Prior calibration experience is required.", arm)
        for i, row in enumerate(rows(queries)):
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            observations.append({arm: query.run(row["text"], arm) for arm in order})
            if (i + 1) % 20 == 0:
                print(f"Completed {i+1} private development requests", flush=True)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       code_sha256=digest(Path(__file__)), **summarize(observations)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
