"""Measure demand-driven prompt construction on the CoF KDD frontier."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import cof_live_certified_batching as paired
from experiments.retrieval.cof_shared_frontier import SUPPORT, panel

ARMS = ("deferred_cached", "shared_frontier", "lazy_shared_frontier")


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
    for offset, qid in enumerate(chosen):
        order = ARMS[offset % len(ARMS):] + ARMS[:offset % len(ARMS)]
        row = {arm: measure(texts[qid], SUPPORT, arm) for arm in order}
        eager_pool, eager = row["shared_frontier"]
        lazy_pool, lazy = row["lazy_shared_frontier"]
        fields = ("selected", "margins", "calls", "padded_tokens", "scored_documents", "rechecked_pairs")
        if eager_pool != lazy_pool or any(eager[field] != lazy[field] for field in fields):
            raise AssertionError("Lazy prompts changed live scoring or selection")
        rows.append((labels[qid], {arm: result for arm, (_, result) in row.items()}))
        if (offset + 1) % 30 == 0:
            print(f"Completed {offset + 1} of {len(chosen)} queries", flush=True)

    report = {}
    for arm in ARMS:
        measurements = [row[arm] for _, row in rows]
        report[arm] = dict(
            latency_p50_p95_ms=np.quantile([result["ms"] for result in measurements], [.50, .95]).tolist(),
            mean_ms=mean(result["ms"] for result in measurements),
            ms_saved_vs_eager=paired_interval([r["shared_frontier"]["ms"] - r[arm]["ms"] for _, r in rows]),
            exact_output_vs_eager=sum(r["shared_frontier"]["selected"] == r[arm]["selected"] for _, r in rows),
            hard_p5=mean(sum(label.get(expert, 0) > 0 for expert, _ in row[arm]["selected"]) / paired.TOP
                         for label, row in rows),
            reviewer_p5_with_citation=mean(sum(label.get(expert, 0) > 0 and bool(docs)
                                              for expert, docs in row[arm]["selected"]) / paired.TOP
                                           for label, row in rows),
            **{name: mean(result[name] for result in measurements) for name in
               ("candidate_documents", "tokenized_documents", "calls", "rechecked_pairs", "padded_tokens")},
        )
    write_json(args.output, dict(
        queries=len(chosen), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), inputs=inputs,
        selected_panel_sha256=sha256("\n".join(chosen).encode()).hexdigest(), support=SUPPORT,
        script_sha256=digest(Path(__file__)), runner_sha256=digest(Path(paired.__file__)),
        embedding_revision=encoder["revision"], reranker_revision=reader["revision"], report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(parser.parse_args())
