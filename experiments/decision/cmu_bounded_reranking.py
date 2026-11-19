"""Live early stopping for bounded dense and cross-encoder evidence scores."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.cmu_baseline import text
from expert_search.cmu_gold import load_annotations, pairwise_loss
from expert_search.retrieval import tokens


TASK = ("Determine whether the document demonstrates research methods and subject matter "
        "that would specifically qualify its author to review the query paper.")
BONUS = 0.10
BATCH = 8
ARMS = ("dense", "first_two", "first_five", "cross_only", "full", "bounded", "bounded_upfront")


def read_dataset(root):
    dataset = root / "evaluation_datasets/d_20_1"
    annotations = load_annotations(root / "data/evaluations.csv")
    submissions = json.loads((dataset / "submissions.json").read_bytes())
    profiles, published = {}, {}
    for reviewer in sorted(annotations):
        rows = [json.loads(line) for line in
                (dataset / "archives" / f"~{reviewer}.jsonl").read_text().splitlines() if line.strip()]
        for row in rows:
            document = text(row).strip()
            existing = published.setdefault(row["id"], document)
            if document != existing:
                raise ValueError("Conflicting publication text")
        profiles[reviewer] = sorted({row["id"] for row in rows})
    queries = {pid: text(submissions[pid]).strip() for grades in annotations.values() for pid in grades}
    titles = {pid: tuple(tokens(document.split("\n", 1)[0])) for pid, document in published.items()}
    cases = []
    for reviewer, grades in sorted(annotations.items()):
        for pid in sorted(grades):
            title = tuple(tokens(queries[pid].split("\n", 1)[0]))
            owned = tuple(doc for doc in profiles[reviewer] if doc != pid and titles[doc] != title)
            if not title or not owned:
                raise ValueError("Missing title or profile after self-match removal")
            cases.append((reviewer, pid, owned))
    return cases, annotations, queries, published


def evidence_order(dense, score_margins, arm):
    """Return the best seen source and keep the dense/index tie order stable."""
    if arm == "dense":
        return 0, float(dense[0]), 0, 0, 0
    limit = min(len(dense), {"first_two": 2, "first_five": 5}.get(arm, len(dense)))
    upper = dense + BONUS
    cursor = 0
    if arm == "bounded_upfront":
        # The best dense score is a lower bound even before checking any source.
        limit = int(np.searchsorted(-upper, -dense[0] + 1e-12, side="right"))
    clipped = calls = 0
    best = -math.inf
    winner = None
    while cursor < limit:
        end = min(limit, cursor + (2 if arm == "bounded" and cursor == 0 else BATCH))
        if arm in {"bounded", "bounded_upfront"}:
            # Earlier rows win ties. Keep this comparison conservative for rounding.
            while end > cursor and upper[end - 1] + 1e-12 < best:
                end -= 1
            if end == cursor:
                break
        margins, truncated = score_margins(cursor, end)
        margins = np.asarray(margins, dtype=np.float64)
        if margins.shape != (end - cursor,) or not np.isfinite(margins).all():
            raise ValueError("Unexpected verifier margins")
        scores = margins if arm == "cross_only" else (
            dense[cursor:end] + BONUS * np.exp(-np.logaddexp(0., -margins))
        )
        choice = int(scores.argmax())
        if scores[choice] > best:
            winner, best = cursor + choice, float(scores[choice])
        clipped += truncated
        calls += 1
        cursor = end
    return winner, best, cursor, calls, clipped


def run(root, embedding, reranker, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_reranking import write_json
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and a Slurm CUDA allocation")
    cases, annotations, queries, published = read_dataset(root)
    projection = json.dumps([cases, annotations, queries, published], sort_keys=True, ensure_ascii=False)
    encoder_meta = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    started = perf_counter()
    ids = sorted(published)
    vectors = encoder.encode([published[doc] for doc in ids])
    by_doc = dict(zip(ids, vectors, strict=True))
    resident = {owned: np.stack([by_doc[doc] for doc in owned]) for owned in {row[2] for row in cases}}
    index_seconds = perf_counter() - started
    encoder.batch_size = 1

    def request(pid, owned, arm):
        start = perf_counter()
        query = queries[pid]
        dense = (resident[owned] @ encoder.encode([query], query=True)[0]).astype(np.float64)
        order = np.argsort(-dense, kind="stable")

        def check(left, right):
            return verifier.score_pairs([(query, published[owned[row]]) for row in order[left:right]], task=TASK)

        index, value, pairs, calls, clipped = evidence_order(dense[order], check, arm)
        return dict(score=value, witness=owned[order[index]], pairs=pairs, calls=calls,
                    clipped=clipped, ms=1000 * (perf_counter() - start))

    for arm in ARMS:
        request(*cases[0][1:], arm)
    rows = []
    for i, (_, pid, owned) in enumerate(cases):
        offset = i % len(ARMS)
        rows.append({arm: request(pid, owned, arm) for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 100 == 0:
            print(i + 1, "cases", flush=True)
    results, all_predictions = {}, {}
    for arm in ARMS:
        measurements = [row[arm] for row in rows]
        predictions = {reviewer: {} for reviewer in annotations}
        for (reviewer, pid, _), measured in zip(cases, measurements, strict=True):
            predictions[reviewer][pid] = measured["score"]
        all_predictions[arm] = predictions
        results[arm] = pairwise_loss(predictions, annotations) | dict(
            p50_p95_ms=np.quantile([row["ms"] for row in measurements], [.50, .95]).tolist(),
            mean_checks=mean(row["pairs"] for row in measurements),
            mean_calls=mean(row["calls"] for row in measurements),
            truncated_checks=sum(row["clipped"] for row in measurements),
            witness_matches_full=sum(row[arm]["witness"] == row["full"]["witness"] for row in rows),
            max_score_difference_from_full=max(abs(row[arm]["score"] - row["full"]["score"]) for row in rows),
        )
    for arm in ARMS:
        predictions = all_predictions[arm]
        reference = all_predictions["full"]
        results[arm]["comparable_order_disagreements_vs_full"] = sum(
            int(np.sign(scores[x] - scores[y]) != np.sign(reference[reviewer][x] - reference[reviewer][y]))
            for reviewer, scores in predictions.items()
            for x, y in combinations(scores, 2) if annotations[reviewer][x] != annotations[reviewer][y]
        )
    write_json(output, dict(
        projection_sha256=sha256(projection.encode()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        embedding_revision=encoder_meta["revision"], reranker_revision=verifier.provenance["revision"],
        bonus=BONUS, max_batch=BATCH, reviewers=len(annotations), cases=len(cases),
        distinct_queries=len(queries), publications=len(published), index_seconds=index_seconds,
        results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
