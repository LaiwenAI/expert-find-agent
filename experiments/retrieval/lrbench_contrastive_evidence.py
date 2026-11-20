"""Test anonymous, order-reversed source comparisons for LR-Bench reviewer choice."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import lrbench_matched_profiles as matched
from experiments.retrieval.lrbench_budgeted_evidence import partitions, threshold
from experiments.retrieval.lrbench_frozen_baselines import paired_stats, read_split
from experiments.retrieval.lrbench_source_facets import TASKS, normalized_title
from experiments.transfer.constraint_counterfactual import Judge


ARMS = ("pointwise", "one_order", "both_orders")
DEPTH = 2
TASK = (
    "Choose which anonymous candidate's supplied publications give stronger evidence of "
    "familiarity with the target paper's research question and technical methods. "
    "Judge only the evidence supplied. Ignore publication count, author prestige, and the "
    "order of the candidates. Output only 0 for candidate 0, 1 for candidate 1, "
    "or 2 if neither has stronger evidence."
)


def join_inputs(release):
    pairs, reference = read_split(release, "pc")
    rated, replay = matched.read_inputs(release)["pc"]
    if reference != replay:
        raise ValueError("Misaligned LR-Bench panel")
    joined = []
    bank = {}
    for pair, prepared in zip(pairs, rated, strict=True):
        matched_pair = []
        for (query, documents), (input_query, profile, encoded_documents) in zip(pair, prepared, strict=True):
            history = []
            for document, encoded in zip(documents, encoded_documents, strict=True):
                if normalized_title(document) == normalized_title(query):
                    continue
                if encoded in bank and bank[encoded] != document:
                    raise ValueError("Inconsistent raw document surface")
                bank[encoded] = document
                history.append(encoded)
            if not history:
                raise ValueError("Missing nonoverlapping publication history")
            matched_pair.append((query, input_query, profile, tuple(history)))
        if matched_pair[0][:2] != matched_pair[1][:2]:
            raise ValueError("Expected one target per paper-centric comparison")
        joined.append(matched_pair)
    return joined, reference, bank


def make_index(encoder, cases, documents):
    profiles = list(dict.fromkeys(profile for pair in cases for _, _, profile, _ in pair))
    sources = list(documents)
    return (
        dict(zip(profiles, encoder.encode(profiles, mean_pool=True), strict=True)),
        dict(zip(sources, encoder.encode(sources, mean_pool=True), strict=True)),
    )


def nominate(pair, encoder, index, documents):
    start = perf_counter()
    profiles, bank = index
    vector = encoder.encode([pair[0][1]], mean_pool=True)[0]
    compact = tuple(float(profiles[profile] @ vector) for _, _, profile, _ in pair)
    compact_ms = (perf_counter() - start) * 1000
    lists = []
    raw = []
    for _, _, _, history in pair:
        scores = np.stack([bank[doc] for doc in history]) @ vector
        order = np.argsort(-scores, kind="stable")
        lists.append(tuple(documents[history[i]] for i in order[:DEPTH]))
        raw.append(float(scores[order[:3]].mean()))
    return compact, tuple(raw), lists, compact_ms, (perf_counter() - start) * 1000


def pointwise(verifier, query, lists):
    pairs = [(query, doc) for entries in lists for doc in entries]
    scores, clipped = verifier.score_pairs(pairs, task=TASKS["general"])
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid pointwise evidence scores")
    split = len(lists[0])
    return max(scores[:split]), max(scores[split:])


def first_order(row):
    left, right = sorted((str(row["positive_id"]), str(row["negative_id"])))
    seed = sha256(f"lrbench-evidence-order-v1:{row['anchor_id']}:{left}:{right}".encode()).digest()
    first = (left, right)[seed[0] % 2]
    return (0, 1) if first == str(row["positive_id"]) else (1, 0)


def compare(judge, query, lists, orders):
    torch = judge.torch
    inputs = []
    for lhs, rhs in orders:
        data = dict(target=query, candidate_0=lists[lhs], candidate_1=lists[rhs])
        ids = judge.tokenizer.apply_chat_template(
            [{"role": "system", "content": TASK},
             {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
            tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True,
        )["input_ids"][0]
        inputs.append(dict(input_ids=ids.tolist()))
    payload = judge.tokenizer.pad(inputs, padding=True, return_tensors="pt").to("cuda")
    width = payload["input_ids"].shape[1]
    if width > 4096:
        raise ValueError("Comparison exceeded the predeclared input budget")
    with torch.inference_mode():
        scores = judge.model(**payload, use_cache=False, logits_to_keep=1).logits[:, -1, judge.ids[:3]]
        scores = scores.float().cpu().tolist()
    results = []
    for (lhs, rhs), logits in zip(orders, scores, strict=True):
        chosen = int(np.argmax(logits))
        winner = (lhs, rhs, None)[chosen]
        margin = (logits[0] - logits[1]) * (1 if lhs == 0 else -1)
        results.append((winner, margin))
    return results, payload["input_ids"].numel()


def signed(winner):
    return (0., 0.) if winner is None else ((1., 0.) if winner == 0 else (0., 1.))


def collect(cases, reference, docs, encoder, index, verifier, judge, offset):
    records = []
    for i, (pair, row) in enumerate(zip(cases, reference, strict=True)):
        compact, raw, evidence, compact_ms, common = nominate(pair, encoder, index, docs)
        order = first_order(row)
        query = pair[0][0]
        result = dict(compact=compact, raw_top_three=raw, compact_ms=compact_ms, evidence_ms=common,
                      source_pairs=sum(map(len, evidence)))
        shift = (i + offset) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            if arm == "pointwise":
                result[arm] = pointwise(verifier, query, evidence)
            elif arm == "one_order":
                attempts, slots = compare(judge, query, evidence, [order])
                result[arm] = signed(attempts[0][0])
                result["one_slots"] = slots
            else:
                attempts, slots = compare(judge, query, evidence, [order, order[::-1]])
                left, right = (item[0] for item in attempts)
                result["two_slots"] = slots
                result["order_agreed"] = left == right and left is not None
                result["order_conflict"] = left is not None and right is not None and left != right
                result["both_unknown"] = left is None and right is None
                result["agree_only"] = signed(left if result["order_agreed"] else None)
                result["symmetric_margin"] = (sum(item[1] for item in attempts) / 2, 0.)
            result[f"{arm}_ms"] = common + (perf_counter() - start) * 1000
        records.append(result)
    return records


def summarize(evaluator, reference, rows):
    point = ("compact", "raw_top_three", "pointwise", "one_order", "agree_only", "symmetric_margin")
    scores = {arm: np.array([row[arm] for row in rows]) for arm in point}
    durations = {"compact": np.array([row["compact_ms"] for row in rows]),
                 "raw_top_three": np.array([row["evidence_ms"] for row in rows])}
    for arm, timed in (("pointwise", "pointwise"), ("one_order", "one_order"),
                       ("agree_only", "both_orders"), ("symmetric_margin", "both_orders")):
        durations[arm] = np.array([row[f"{timed}_ms"] for row in rows])
    likelihood = np.abs(scores["compact"][:, 0] - scores["compact"][:, 1])
    usage = {}
    for name, budget in (("agree_fallback_25", .25), ("agree_fallback_50", .5), ("agree_fallback_all", 1.)):
        acquire = np.ones(len(rows), dtype=bool)
        if budget < 1:
            for train, test in partitions(reference):
                acquire[test] = threshold(-likelihood, train, test, budget)
        decisive = (scores["agree_only"][:, 0] != scores["agree_only"][:, 1]) & acquire
        scores[name] = np.where(decisive[:, None], scores["agree_only"], scores["compact"])
        durations[name] = np.where(acquire, durations["agree_only"], durations["compact"])
        usage[name] = dict(
            acquired=int(acquire.sum()),
            source_abstracts=sum(row["source_pairs"] for row, use in zip(rows, acquire, strict=True) if use),
            judge_token_slots=sum(row["two_slots"] for row, use in zip(rows, acquire, strict=True) if use),
            decisive=int(decisive.sum()),
        )

    predictions = {"rate": reference, **{
        name: [dict(row, positive_score=float(left), negative_score=float(right))
               for row, (left, right) in zip(reference, values, strict=True)]
        for name, values in scores.items()
    }}
    metrics = paired_stats(evaluator, "pc", predictions)
    baseline = scores["compact"][:, 0] > scores["compact"][:, 1]
    ids = [row["anchor_id"] for row in reference]
    for arm, values in scores.items():
        change = (values[:, 0] > values[:, 1]).astype(int) - baseline
        metrics[arm].update(
            gain=int((change > 0).sum()), loss=int((change < 0).sum()),
            accuracy_delta_vs_compact=clustered_paired_interval(change, ids),
            p95_ms=float(np.quantile(durations[arm], .95)),
            **usage.get(arm, {}),
        )
    counts = Counter(order_agreed=sum(row["order_agreed"] for row in rows),
                     order_conflict=sum(row["order_conflict"] for row in rows),
                     both_unknown=sum(row["both_unknown"] for row in rows),
                     one_judge_slots=sum(row["one_slots"] for row in rows),
                     two_judge_slots=sum(row["two_slots"] for row in rows),
                     source_abstracts=sum(row["source_pairs"] for row in rows))
    return dict(comparisons=len(rows), anchors=len(set(ids)), order_and_work=counts, metrics=metrics)


def main():
    import torch
    from experiments.followir.core17_verifier import Reranker

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pair, ref, docs = join_inputs(args.release)
    rate = Path("artifacts/models/rate-0.6b")
    pinned = matched.verify_model(rate, "rate06")
    encoder = QwenTorchEmbedding(rate, max_length=2048, batch_size=32)
    encoder.model.config.use_cache = False
    verifier = Reranker(Path("artifacts/models/qwen3-reranker-0.6b"))
    judge = Judge()
    judge.tokenizer.padding_side = "left"
    index = make_index(encoder, pair, docs)
    collect(pair[:1], ref[:1], docs, encoder, index, verifier, judge, 0)
    rows = collect(pair, ref, docs, encoder, index, verifier, judge, args.order_offset)
    spec = importlib.util.spec_from_file_location("rate_evaluator", args.release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    results = summarize(evaluator, ref, rows)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), assets=matched.ASSETS, profile_revision=pinned["revision"],
        pointwise_revision=verifier.provenance["revision"], pairwise_provenance_sha256=judge.sha,
        source_depth=DEPTH, indexed_documents=len(docs), results=results,
    ))


if __name__ == "__main__":
    main()
