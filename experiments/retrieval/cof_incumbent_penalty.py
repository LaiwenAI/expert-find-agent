"""Compare fixed MMR penalties with the incumbent-residual challenger at equal work."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.retrieval import cof_global_profile_pooling as base
from experiments.retrieval.cof_contrastive_swap import change_stats
from experiments.retrieval.cof_residual_challenger import DEPTH, POOL, TOP, ResidualIndex, evaluate


FIXED = {"penalty_025": .25, "penalty_050": .50}
ARMS = ("sixth", *FIXED, "residual_nomination")


class PenaltyIndex(ResidualIndex):
    def nominate(self, primary, arm):
        if arm not in FIXED:
            return super().nominate(primary, arm)
        _, similarities, reviews, order = primary
        first = [(owner, order[owner][:DEPTH]) for owner in reviews[:TOP]]
        depth = len(first[-1][1])
        anchor = base.unit(self.reference.vectors[order[reviews[TOP-1]][0]])
        options = [owner for owner in reviews[TOP:POOL] if len(order[owner]) >= depth]

        def best(owner):
            docs = np.asarray(order[owner])
            overlap = self.reference.vectors[docs] @ anchor
            return float((similarities[docs] - FIXED[arm] * overlap).max())

        if options:
            winner = max(options, key=best)
            first.append((winner, order[winner][:depth]))
        return self.as_ids(first)


def run(embedding, reranker, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = base.verify_checkpoint(embedding)
    root, papers, _, ref, ids, _, truth = base.read_data("SIGIR", provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    encoder = base.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    index = PenaltyIndex(ref)
    warm = index.primary(texts[ids[0]], encoder)
    for arm in ARMS:
        evaluate(index, warm, texts[ids[0]], texts, reader, arm, 0)
    outcomes = {arm: [] for arm in ARMS}
    coefficients = []
    for i, need in enumerate(ids):
        started = perf_counter()
        primary = index.primary(texts[need], encoder)
        elapsed = (perf_counter() - started) * 1000
        vector, _, reviews, order = primary
        anchor = base.unit(ref.vectors[order[reviews[TOP-1]][0]])
        cosine = np.clip(vector @ anchor, 0., 1.)
        coefficients.append(float(cosine / (1 + np.sqrt(1-cosine**2))))
        shift = i % len(ARMS)
        for arm in ARMS[shift:]+ARMS[:shift]:
            outcomes[arm].append(evaluate(index, primary, texts[need], texts, reader, arm, elapsed))
    report = {}
    residual = outcomes["residual_nomination"]
    for arm, rows in outcomes.items():
        counts, difference = Counter(), []
        for i, (row, other) in enumerate(zip(rows, residual, strict=True)):
            hard = truth[ids[i]]
            if row["reader_pairs"] != other["reader_pairs"] or row["ranking"][:4] != other["ranking"][:4]:
                raise AssertionError("Changed verification budget or protected slots")
            hits, gain, loss, _ = change_stats(other, row, hard, with_citations=False)
            cited, _, _, _ = change_stats(other, row, hard, with_citations=True)
            counts.update(
                known_positive=hits, cited_positive=cited,
                differs_from_residual=int(row["nominee"] != other["nominee"]),
                swapped=int(row["swapped"]), reader_pairs=row["reader_pairs"],
                clipped=row["clipped"],
            )
            difference.append((gain-loss)/TOP)
        report[arm] = dict(**counts, p95_ms=float(np.quantile([r["complete_ms"] for r in rows], .95)),
                           paired_p5_delta_vs_residual=base.paired_interval(difference))
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(ids),
        code_sha256=base.digest(Path(__file__)), hard_label_sha256=base.digest(root/"SIGIR_queries_test_hard.json"),
        vector_protocol_sha256=base.digest(root/"qwen-embedding-torch-v1/protocol.json"),
        median_residual_penalty=float(np.median(coefficients)), residual_penalties=[min(coefficients),max(coefficients)],
        arms=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.embedding, args.reranker, args.output)
