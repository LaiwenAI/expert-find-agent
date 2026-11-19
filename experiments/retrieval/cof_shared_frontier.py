"""Evaluate shared optimistic citations with live models on a disjoint CoF KDD slice."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import cof_live_certified_batching as paired
from experiments.retrieval.cof_compact_profile_shortlist import title_tokens

ARMS = ("exhaustive", "guarded_cached", "guarded_bucketed", "deferred_cached",
        "shared_frontier", "cost_aware_frontier")
SUPPORT = 2


def panel(ids, texts):
    excluded = set()
    for name in ("NIPS", "SIGIR", "SciRepEval"):
        _, papers, _, labels = load(name)
        excluded.update(tuple(title_tokens(papers[qid]["title"])) for qid in labels["hard"])
    eligible = [qid for qid in ids if tuple(title_tokens(texts[qid].split("\n", 1)[0])) not in excluded]
    order = sorted(eligible, key=lambda qid: sha256(f"kdd-microbatch-v1:{qid}".encode()).digest())
    if len(eligible) != 170:
        raise ValueError("Changed disjoint KDD panel")
    return order[48:]


def per_request(measure, text, labels, arm):
    ordered, result = measure(text, SUPPORT, arm)
    rows = result["selected"]
    return dict(candidates=[e for e, _ in ordered], response=rows,
                p5=sum(labels.get(e, 0) > 0 for e, _ in rows) / paired.TOP,
                evidence_backed_p5=sum(labels.get(e, 0) > 0 and bool(docs) for e, docs in rows) / paired.TOP,
                cited_slots=sum(bool(docs) for _, docs in rows),
                **{key: result[key] for key in ("ms", "calls", "scored_documents", "rechecked_pairs", "padded_tokens")})


def summary(rows):
    result = {}
    for arm in ARMS:
        arms = [row[arm] for row in rows]
        reference = [row["exhaustive"] for row in rows]
        result[arm] = dict(
            exact_candidate_order=sum(a["candidates"] == b["candidates"] for a, b in zip(arms, reference, strict=True)),
            exact_order_and_citations=sum(a["response"] == b["response"] for a, b in zip(arms, reference, strict=True)),
            p50_p95_ms=np.quantile([a["ms"] for a in arms], [.50, .95]).tolist(),
            **{name: mean(a[name] for a in arms) for name in
               ("p5", "evidence_backed_p5", "cited_slots", "calls", "scored_documents", "rechecked_pairs", "padded_tokens")},
            p5_delta_vs_exhaustive=paired_interval([a["p5"] - b["p5"] for a, b in zip(arms, reference, strict=True)]),
            ms_saved_vs_deferred=paired_interval([row["deferred_cached"]["ms"] - row[arm]["ms"] for row in rows]),
        )
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.dataset = "KDD"
    args.exclude_nips_titles = False
    measure, ids, texts, labels, inputs, encoder, reader = paired.prepare_runner(args)
    chosen = panel(ids, texts)
    warmup = next(qid for qid in ids if qid not in chosen)
    for arm in ARMS:
        measure(texts[warmup], SUPPORT, arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: per_request(measure, texts[qid], labels[qid], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1} of {len(chosen)} requests", flush=True)
    write_json(args.output, dict(
        queries=len(chosen), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), inputs=inputs,
        selected_panel_sha256=sha256("\n".join(chosen).encode()).hexdigest(), support=SUPPORT,
        script_sha256=digest(Path(__file__)), runner_sha256=digest(Path(paired.__file__)),
        algorithm_sha256=digest(Path(paired.resolve_shared_frontier.__code__.co_filename)),
        embedding_revision=encoder["revision"], reranker_revision=reader["revision"], arms=summary(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(parser.parse_args())
