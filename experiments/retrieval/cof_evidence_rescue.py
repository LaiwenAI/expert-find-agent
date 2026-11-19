"""Locate and test the conversion of released CoF positives into cited results."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.deferred_verification import _extreme, resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, select
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP, WIDTH, WaveReader
from experiments.retrieval.cof_live_prefix_stop import TASK


ARMS = {
    "dense_attach3": ("dense", 3, False, None),
    "blend_attach3": ("mean_max", 3, False, None),
    "blend_attach6": ("mean_max", 6, False, None),
    "blend_promote3": ("mean_max", 3, True, None),
    "blend_budget6": ("mean_max", 6, True, 15),
}
SUPPORT = 2


def budgeted_frontier(ordered, reader, budget):
    """Stop at a hard scoring cap and return only verified support."""
    owners = {}
    for expert, sources in ordered:
        for doc in sources:
            owners.setdefault(doc, []).append(expert)
    definite, provisional = {}, set()
    spent = 0
    while True:
        lower, _ = _extreme(ordered, definite, SUPPORT, TOP, False)
        upper, _ = _extreme(ordered, definite, SUPPORT, TOP, True)
        if lower == upper or spent == budget:
            return lower, spent, lower == upper
        pending = {}
        for expert, docs in upper:
            for doc in docs:
                if (expert, doc) not in definite:
                    pending.setdefault(doc, expert)
        unseen = [doc for doc in pending if doc not in provisional]
        docs = (unseen or list(pending))[:min(WIDTH, budget - spent)]
        if not docs:
            raise AssertionError("Unresolved output without a pending source")
        scorer = reader.score if unseen else reader.recheck
        values = scorer([(pending[doc], doc) for doc in docs])
        spent += len(docs)
        for doc, value in zip(docs, values, strict=True):
            provisional.add(doc)
            if not unseen or abs(value) > GUARD_MARGIN:
                for expert in owners[doc]:
                    definite[expert, doc] = value


def run_request(index, encoder, verifier, texts, query, labels, arm):
    route, depth, promote, cap = ARMS[arm]
    start = perf_counter()
    hits = index.search_query(query, encoder, route, depth=depth)
    retrieved = perf_counter()
    pool = hits if promote else hits[:TOP]
    reader = WaveReader(verifier, query, texts, {doc for _, docs in pool for doc in docs},
                        dynamic=True, cache=True)
    if cap is None:
        output = resolve_shared_frontier(
            pool, reader.score, reader.recheck, reader.length, support=SUPPORT, top=TOP, width=WIDTH,
            band=GUARD_MARGIN, cost_aware=False,
        )["selected"]
        certified = True
    else:
        output, spent, certified = budgeted_frontier(pool, reader, cap)
        if spent != reader.scored_documents + reader.rechecked_pairs:
            raise AssertionError("Scoring budget does not match actual calls")
    if not promote:
        by_owner = dict(output)
        output = [(expert, by_owner[expert]) for expert, _ in pool]
    finished = perf_counter()
    known = {expert for expert, grade in labels.items() if grade > 0}
    head = {expert for expert, _ in hits[:TOP]}
    positive = {expert for expert, _ in output} & known
    cited = {expert for expert, docs in output if docs} & known
    return dict(
        p5=len(positive)/TOP, cited_p5=len(cited)/TOP, promoted_positives=len(positive-head),
        displaced_positives=len((head & known)-positive),
        explicit_negative_cited=sum(labels.get(expert) == 0 for expert, docs in output if docs),
        cited_unjudged=sum(expert not in labels for expert, docs in output if docs),
        checked=reader.scored_documents+reader.rechecked_pairs, certified=certified,
        retrieval_ms=1000*(retrieved-start), complete_ms=1000*(finished-start),
        original_positive_top10=len(known & {expert for expert, _ in hits}),
        original_positive_top5=len(known & head),
    )


def diagnose(index, encoder, verifier, texts, query, labels):
    # Audit every retrieved known-positive candidate, never use these labels to route.
    dense = index.search_query(query, encoder, "dense")
    blend = index.search_query(query, encoder, "mean_max", depth=6)
    seen = {expert for expert, _ in dense}
    groups = {key: Counter() for key in (
        "blend_only_top5", "blend_only_ranks6_10", "shared_top5", "shared_ranks6_10")}
    docs = sorted({doc for expert, sources in blend if labels.get(expert) == 1 for doc in sources})
    margins = {}
    for offset in range(0, len(docs), WIDTH):
        batch = docs[offset:offset+WIDTH]
        values, _ = verifier.score_pairs([(query, texts[doc]) for doc in batch], task=TASK)
        margins.update(zip(batch, values, strict=True))
    for i, (expert, sources) in enumerate(blend):
        if labels.get(expert) != 1:
            continue
        name = ("shared" if expert in seen else "blend_only") + ("_top5" if i < TOP else "_ranks6_10")
        group = groups[name]
        group["positives"] += 1
        group["fewer_than_two_sources"] += len(sources) < SUPPORT
        count = sum(margins[doc] >= 0 for doc in sources[:3])
        expanded = sum(margins[doc] >= 0 for doc in sources)
        group["supported_by3"] += count >= SUPPORT
        group["supported_by6"] += expanded >= SUPPORT
        group["zero_positive_by6"] += expanded == 0
    return groups


def report(rows):
    result = {}
    for arm in ARMS:
        sample = [row[arm] for row in rows]
        result[arm] = dict(
            **{field: mean(row[field] for row in sample) for field in ("p5", "cited_p5", "checked")},
            **{field: sum(row[field] for row in sample) for field in
               ("promoted_positives", "displaced_positives", "explicit_negative_cited", "cited_unjudged",
                "certified", "original_positive_top10", "original_positive_top5")},
            **{field: np.quantile([row[field] for row in sample], [.5, .95]).tolist()
               for field in ("retrieval_ms", "complete_ms")},
            cited_p5_delta=paired_interval([row[arm]["cited_p5"]-row["blend_attach3"]["cited_p5"] for row in rows]),
        )
    return result


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "")+"\n"+paper.get("abstract", "")).strip() for pid, paper in papers.items()}
    index = ProfileIndex(reference)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    arms = tuple(ARMS)
    for arm in arms:
        run_request(index, encoder, verifier, texts, texts[warm], labels[warm], arm)
    cases, diagnostic = [], {}
    for i, q in enumerate(chosen):
        offset = i % len(arms)
        cases.append({arm: run_request(index, encoder, verifier, texts, texts[q], labels[q], arm)
                      for arm in arms[offset:]+arms[:offset]})
        for key, counts in diagnose(index, encoder, verifier, texts, texts[q], labels[q]).items():
            diagnostic.setdefault(key, Counter()).update(counts)
        if (i+1) % 30 == 0:
            print("Completed queries:", i+1, flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(cases), cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        script_sha256=digest(Path(__file__)), index_script_sha256=digest(Path(ProfileIndex.__init__.__code__.co_filename)),
        hard_label_sha256=digest(root/f"{dataset}_queries_test_hard.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=report(cases), diagnostic=diagnostic,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
