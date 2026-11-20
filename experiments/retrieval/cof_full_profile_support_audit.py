"""Audit verifier support across the complete histories of retrieved CoF positives."""

import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_profile_pooling import select
from experiments.retrieval.cof_live_certified_batching import CAP, POOL, TOP, WIDTH, WaveReader

DEPTHS = (3, 6, 12)


def summarize_profile(ordered, verdicts):
    accepted = [i for i, source in enumerate(ordered, 1) if verdicts[source] >= 0]
    second = accepted[1] if len(accepted) >= 2 else None
    return Counter({
        "positives": 1,
        "fewer_than_two_catalog_sources": int(len(ordered) < 2),
        "zero_accepted_in_catalog": int(not accepted),
        "one_accepted_in_catalog": int(len(accepted) == 1),
        "two_accepted_in_catalog": int(second is not None),
        **{f"two_accepted_at_{depth}": int(second is not None and second <= depth) for depth in DEPTHS},
    })


def audit_query(index, verifier, texts, vector, query, truth, observe=None):
    reference = index.reference
    device_query = torch.as_tensor(vector, dtype=torch.float32, device="cuda")
    scores = torch.mv(index.vectors, device_query).cpu().numpy()
    maxima = np.maximum.reduceat(scores[reference.owner_rows], reference.owner_starts)
    order = np.argsort(-maxima, kind="stable")[:POOL]
    positions = [(rank, reference.owner_ids[i]) for rank, i in enumerate(order)
                 if truth.get(reference.owner_ids[i], 0) > 0]
    results = {name: Counter() for name in ("ranks_1_to_5", "ranks_6_to_10")}
    if not positions:
        return results, 0, 0
    profiles = {
        owner: [reference.source_ids[i] for i in sorted(
            reference.owners[owner], key=lambda i: (-float(scores[i]), reference.source_ids[i]),
        )] for _, owner in positions
    }
    distinct = sorted({doc for sources in profiles.values() for doc in sources})
    reader = WaveReader(verifier, query, texts, distinct, cache=False)
    # Pin every scoring call to the same numerical shape as every other call.
    # The request benchmarks use other shapes and are reported separately.
    reader.maximum = CAP
    accepted = {}
    for offset in range(0, len(distinct), WIDTH):
        chunk = distinct[offset:offset+WIDTH]
        values = reader.recheck([("", doc) for doc in chunk])
        accepted.update(zip(chunk, values, strict=True))
    for rank, owner in positions:
        name = "ranks_1_to_5" if rank < TOP else "ranks_6_to_10"
        results[name].update(summarize_profile(profiles[owner], accepted))
        if observe is not None and rank < TOP:
            observe(owner, profiles[owner], accepted)
    return results, len(distinct), reader.truncated


@torch.inference_mode()
def run(args, *, observer=None):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Use a fresh output on Slurm CUDA")
    start = perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(args.embedding)
    root, papers, _, reference, eligible, _, labels = read_data(args.dataset, provenance)
    texts = {key: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for key, paper in papers.items()}
    tested = select(args.dataset, eligible, texts)
    index = GpuOwnedVectorIndex(reference)
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    verifier = Reranker(args.reranker)
    groups = {name: Counter() for name in ("ranks_1_to_5", "ranks_6_to_10")}
    pairs = truncated = 0
    for i, qid in enumerate(tested, 1):
        query = texts[qid]
        vector = encoder.encode([query], query=True)[0]
        observe = None if observer is None else partial(observer, qid, texts)
        counts, scored, cut = audit_query(index, verifier, texts, vector, query, labels[qid], observe)
        pairs += scored
        truncated += cut
        for name, values in counts.items():
            groups[name].update(values)
        if i % 20 == 0:
            print(f"Audited {i}/{len(tested)} queries and {pairs} distinct query-source pairs.", flush=True)
    write_json(args.output, dict(
        dataset=args.dataset, queries=len(tested),
        reviewers=len(reference.owner_ids), sources=len(reference.source_ids),
        cohort_sha256=sha256("\n".join(tested).encode()).hexdigest(),
        labels_sha256=digest(root / f"{args.dataset}_queries_test_hard.json"),
        script_sha256=digest(Path(__file__)), groups=groups, scored_pairs=pairs, truncated_pairs=truncated,
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        score_batch_size=WIDTH, score_sequence_cap=CAP, seconds=perf_counter()-start,
        gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
