"""Transfer a frozen two-order evidence comparison to CoF top-five reviewer search."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.retrieval import cof_global_profile_pooling as base
from experiments.retrieval.cof_competing_profile_evidence import SUPPORT_MARGIN, TASK
from experiments.retrieval.lrbench_contrastive_evidence import compare
from experiments.transfer.constraint_counterfactual import Judge


TOP = 5
EVIDENCE = 2
ARMS = ("dense_five", "pointwise_swap", "one_order", "both_orders", "deferred_reverse")


def first_order(query_id, hits):
    pair = sorted((str(hits[TOP - 1][0]), str(hits[TOP][0])))
    seed = sha256(f"cof-evidence-order-v1:{query_id}:{pair[0]}:{pair[1]}".encode()).digest()
    return (0, 1) if pair[seed[0] % 2] == str(hits[TOP - 1][0]) else (1, 0)


def score_sources(verifier, query, hits, texts):
    pairs = [(query, texts[source]) for _, documents in hits for source in documents]
    margins, clipped = verifier.score_pairs(pairs, task=TASK)
    if not np.isfinite(margins).all() or len(margins) != len(pairs):
        raise ValueError("Invalid source scores")
    offset = 0
    rows = []
    for _, documents in hits:
        rows.append(tuple(margins[offset:offset + len(documents)]))
        offset += len(documents)
    return rows, clipped, len(pairs)


def confirmed_swap(judge, query, texts, order, arm):
    if arm == "both_orders":
        votes, tokens = compare(judge, query, texts, [order, order[::-1]])
        return votes[0][0] == votes[1][0] == 1, tokens, 2, 1
    votes, tokens = compare(judge, query, texts, [order])
    if votes[0][0] != 1 or arm == "one_order":
        return votes[0][0] == 1, tokens, 1, 1
    votes, reverse = compare(judge, query, texts, [order[::-1]])
    return votes[0][0] == 1, tokens + reverse, 2, 2


def first_or_swap(arm, hits, query_id, query, texts, verifier, judge):
    start = perf_counter()
    finalists = hits[:TOP]
    token_slots = judge_prompts = judge_calls = 0
    swap = False
    if arm == "pointwise_swap":
        scores, clipped, scored = score_sources(verifier, query, hits, texts)
        swap = max(scores[TOP][:EVIDENCE]) > max(scores[TOP - 1][:EVIDENCE])
        if swap:
            finalists = hits[:TOP - 1] + [hits[TOP]]
            scores = scores[:TOP - 1] + [scores[TOP]]
    else:
        if arm != "dense_five":
            evidence = [tuple(texts[source] for source in documents[:EVIDENCE])
                        for _, documents in hits[TOP - 1:TOP + 1]]
            swap, token_slots, judge_prompts, judge_calls = confirmed_swap(
                judge, query, evidence, first_order(query_id, hits), arm,
            )
            if swap:
                finalists = hits[:TOP - 1] + [hits[TOP]]
        scores, clipped, scored = score_sources(verifier, query, finalists, texts)

    ranking = tuple(owner for owner, _ in finalists)
    citations = {}
    for (owner, sources), margins in zip(finalists, scores[:TOP], strict=True):
        allowed = [source for source, margin in zip(sources, margins, strict=True) if margin > SUPPORT_MARGIN]
        if len(allowed) >= 2:
            citations[owner] = tuple(allowed[:2])
    return dict(
        ranking=ranking, cited=citations, swapped=swap,
        judge_token_slots=token_slots, judge_prompts=judge_prompts, judge_calls=judge_calls,
        citation_pairs=scored, citation_pairs_truncated=clipped,
        extra_ms=(perf_counter() - start) * 1000,
    )


def change_stats(reference, current, truth, *, with_citations):
    if with_citations:
        old, new = set(reference["cited"]), set(current["cited"])
    else:
        old, new = set(reference["ranking"]), set(current["ranking"])
    found = sum(truth.get(owner, 0) > 0 for owner in new)
    added, removed = new - old, old - new
    gained = sum(truth.get(owner, 0) > 0 for owner in added)
    lost = sum(truth.get(owner, 0) > 0 for owner in removed)
    lower = gained - lost - len(removed - truth.keys())
    upper = gained - lost + len(added - truth.keys())
    return found, gained, lost, (lower / TOP, upper / TOP)


def summarize(records, labels):
    reference = [record["dense_five"] for record in records]
    outcomes = {}
    for arm in ARMS:
        totals = Counter()
        rank_diff, cited_diff, bounds, cited_bounds, times = [], [], [], [], []
        for previous, current, truth in zip(reference, (r[arm] for r in records), labels, strict=True):
            if len(current["ranking"]) != TOP or previous["ranking"][:4] != current["ranking"][:4]:
                raise AssertionError("Changed the fixed issuance or four protected recommendations")
            positive, win, loss, bound = change_stats(previous, current, truth, with_citations=False)
            cited, cited_win, cited_loss, cited_bound = change_stats(previous, current, truth, with_citations=True)
            rank_diff.append((win-loss)/TOP)
            cited_diff.append((cited_win-cited_loss)/TOP)
            bounds.append(bound)
            cited_bounds.append(cited_bound)
            times.append(current["complete_ms"])
            totals.update(
                known_positive=positive, known_positive_added=win, known_positive_lost=loss,
                known_cited_positive=cited, known_cited_positive_added=cited_win, known_cited_positive_lost=cited_loss,
                judged_final=sum(owner in truth for owner in current["ranking"]),
                cited=len(current["cited"]), cited_unjudged=len(set(current["cited"]) - truth.keys()),
                **{key: current[key] for key in ("swapped", "judge_token_slots", "judge_prompts", "judge_calls",
                                                 "citation_pairs", "citation_pairs_truncated")},
            )
        outcomes[arm] = dict(
            **totals, known_p5_delta=paired_interval(rank_diff), known_cited_p5_delta=paired_interval(cited_diff),
            missing_label_p5_bounds=[mean(values) for values in zip(*bounds, strict=True)],
            missing_label_cited_p5_bounds=[mean(values) for values in zip(*cited_bounds, strict=True)],
            p50_ms=float(np.quantile(times, .50)), p95_ms=float(np.quantile(times, .95)),
        )
    return outcomes


def run(dataset, embedding, reranker, output, order_offset):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = base.verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, truth = base.read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = base.ProfileIndex(reference)
    encoder = base.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = base.Reranker(reranker)
    judge = Judge()
    judge.tokenizer.padding_side = "left"
    chosen = ids if dataset == "SIGIR" else base.select(dataset, ids, texts)
    warm = index.search_query(texts[chosen[0]], encoder, "mean_max")[:TOP + 1]
    for arm in ARMS:
        first_or_swap(arm, warm, chosen[0], texts[chosen[0]], texts, verifier, judge)
    records, retrieval = [], []
    pair_labels = Counter()
    for i, q in enumerate(chosen):
        start = perf_counter()
        hits = index.search_query(texts[q], encoder, "mean_max")[:TOP + 1]
        elapsed = (perf_counter() - start) * 1000
        retrieval.append(elapsed)
        original, challenger = (truth[q].get(owner) for owner, _ in hits[TOP-1:TOP+1])
        pair_labels.update(
            both_judged=int(original is not None and challenger is not None),
            challenger_positive_incumbent_negative=int(challenger is not None and challenger > 0 and original == 0),
            incumbent_positive_challenger_negative=int(original is not None and original > 0 and challenger == 0),
            challenger_positive_incumbent_unjudged=int(challenger is not None and challenger > 0 and original is None),
        )
        shift = (i + order_offset) % len(ARMS)
        trials = {}
        for arm in ARMS[shift:] + ARMS[:shift]:
            result = first_or_swap(arm, hits, q, texts[q], texts, verifier, judge)
            trials[arm] = dict(result, complete_ms=elapsed+result["extra_ms"])
        records.append(trials)
        if (i + 1) % 30 == 0:
            print(f"{dataset}: {i + 1}/{len(chosen)}", flush=True)
    base.write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(chosen), experts=len(reference.owner_ids), sources=len(reference.source_ids),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        hard_label_sha256=base.digest(root / f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=base.digest(root / "qwen-embedding-torch-v1/protocol.json"),
        code_sha256=base.digest(Path(__file__)),
        comparison_code_sha256=base.digest(Path(compare.__code__.co_filename)),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        comparison_model_sha256=judge.sha, order_offset=order_offset,
        pair_labels=pair_labels, retrieval_p95_ms=float(np.quantile(retrieval, .95)),
        deferred_different_outputs=sum(
            row["both_orders"]["ranking"] != row["deferred_reverse"]["ranking"]
            or row["both_orders"]["cited"] != row["deferred_reverse"]["cited"]
            for row in records
        ),
        arms=summarize(records, [truth[q] for q in chosen]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("SIGIR", "KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.embedding, args.reranker, args.output, args.order_offset)
