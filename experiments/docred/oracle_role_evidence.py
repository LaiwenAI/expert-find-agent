"""Separate within-document evidence selection from role verification on Re-DocRED."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import person_bound_evidence as base
from experiments.qasper.paired_reader import load_reader


WINDOWS = ("matched", "gold", "full")
SYSTEM = base.TASK + " Reply 1 for explicit support or 0 otherwise."


def matched_window(case, count):
    words, _, idf = base.profile(case["sentences"])
    names = [tuple(base.tokens(name)) for name in (*case["people"], case["work"])]
    coverage = [[max(float(base.contains(row, name)), .6 * base.coverage(name, row, idf))
                 for row in words] for name in names]
    role = base.tokens(base.ROLE_CUES[case["role"]])
    coverage.append([base.coverage(role, row, idf) for row in words])

    def key(indices):
        support = [max(signal[i] for i in indices) for signal in coverage]
        return sum(support), min(support[:2]), tuple(-i for i in indices)

    # The short, unrestricted Re-DocRED documents permit exact subset selection.
    return max(combinations(range(len(words)), count), key=key)


def reader_scorer(tokenizer, reader):
    ids = [tokenizer.encode(value, add_special_tokens=False) for value in ("1", "0")]
    if any(len(label) != 1 for label in ids) or ids[0] == ids[1]:
        raise ValueError("Expected distinct single-token decisions")
    labels = [label[0] for label in ids]

    @torch.inference_mode()
    def score(pairs):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Source:\n{doc}\n\nQuestion: {query}"}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ) for query, doc in pairs]
        if max(map(len, prompts)) > reader.config.max_position_embeddings:
            raise ValueError("Reader context overflow")
        inputs = tokenizer.pad([{"input_ids": prompt} for prompt in prompts], padding=True,
                               padding_side="left", return_tensors="pt").to("cuda")
        logits = reader(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, labels].float()
        return (logits[:, 0] - logits[:, 1]).cpu().tolist(), 0

    return score


def measure(cases, score):
    rows = []
    for i, case in enumerate(cases):
        gold = tuple(sorted(case["annotated"]))
        count = len(gold)
        full = tuple(range(len(case["sentences"])))
        matched = matched_window(case, count)
        windows = dict(matched=matched, gold=gold, full=full)
        reverse = int(case["id"][0], 16) % 2
        people = case["people"][::1 - 2 * reverse]
        queries = [f"Was {name} credited in the role of {base.ROLES[case['role']]} for {case['work']}?"
                   for name in people]
        results = {}
        offset = i % len(WINDOWS)
        for arm in WINDOWS[offset:] + WINDOWS[:offset]:
            indices = windows[arm]
            text = "\n".join(case["sentences"][j] for j in indices)
            start = perf_counter()
            margins, clipped = score([(query, text) for query in queries])
            elapsed = (perf_counter() - start) * 1000
            if clipped or not np.isfinite(margins).all():
                raise ValueError("Invalid or truncated paired score")
            results[arm] = dict(margins=margins[::1 - 2 * reverse], ms=elapsed)
        rows.append((case, gold == matched, results))
    return rows


def summary(rows, thresholds):
    reference = [int(result["matched"]["margins"][0] > thresholds["matched"] >= result["matched"]["margins"][1])
                 for _, _, result in rows]
    docs = [case["doc"] for case, _, _ in rows]
    output = {}
    for arm, threshold in thresholds.items():
        results = [result[arm] for _, _, result in rows]
        margins = [row["margins"] for row in results]
        both = [int(positive > threshold >= rival) for positive, rival in margins]
        output[arm] = dict(
            accepted_supported=sum(positive > threshold for positive, _ in margins),
            accepted_rival=sum(rival > threshold for _, rival in margins),
            ordered=sum(positive > rival for positive, rival in margins),
            both_correct=sum(both),
            rescued_vs_matched=sum(a > b for a, b in zip(both, reference, strict=True)),
            lost_vs_matched=sum(a < b for a, b in zip(both, reference, strict=True)),
            both_delta_vs_matched=clustered_paired_interval(np.subtract(both, reference), docs),
            inference_pair_p95_ms=float(np.quantile([row["ms"] for row in results], .95)),
        )
    return output


def paired_model_deltas(reranker_rows, reader_rows, reranker_cuts, reader_cuts):
    docs = [case["doc"] for case, _, _ in reranker_rows]
    if [case["id"] for case, _, _ in reranker_rows] != [case["id"] for case, _, _ in reader_rows]:
        raise ValueError("Misaligned model observations")
    report = {}
    for arm in WINDOWS:
        small = [result[arm]["margins"] for _, _, result in reranker_rows]
        large = [result[arm]["margins"] for _, _, result in reader_rows]
        both_diff, order_diff = [], []
        for (a, b), (c, d) in zip(small, large, strict=True):
            both_diff.append(int(c > reader_cuts[arm] >= d) - int(a > reranker_cuts[arm] >= b))
            order_diff.append(int(c > d) - int(a > b))
        report[arm] = dict(both=clustered_paired_interval(both_diff, docs),
                           order=clustered_paired_interval(order_diff, docs))
    return report


def gold_name_coverage(cases):
    counts = Counter()
    for case in cases:
        words = [tuple(base.tokens(case["sentences"][i])) for i in case["annotated"]]
        supported, rival = (any(base.contains(row, tuple(base.tokens(name))) for row in words)
                            for name in case["people"])
        counts[f"supported_{int(supported)}_rival_{int(rival)}"] += 1
    return dict(sorted(counts.items()))


def run(snapshot, reranker, reader, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    panels, counts, titles = {}, {}, {}
    for split, expected in base.DATA.items():
        path = snapshot / f"{split}_revised.json"
        if base.digest(path) != expected:
            raise ValueError(f"Changed Re-DocRED split: {split}")
        source = json.loads(path.read_bytes())
        titles[split] = {doc["title"] for doc in source}
        panels[split] = [case for case in base.flagged_pairs(source) if case["annotated"]]
        counts[split] = dict(pairs=len(panels[split]), documents=len({case["doc"] for case in panels[split]}),
                             exact_names_in_gold=gold_name_coverage(panels[split]),
                             mean_gold_sentences=float(np.mean([len(case["annotated"]) for case in panels[split]])))
    if titles["dev"] & titles["test"]:
        raise ValueError("Overlapping source documents")
    cap = len(panels["dev"]) // 20
    reranker_model = base.Reranker(reranker)
    tokenizer, reader_model, reader_provenance = load_reader(reader)

    def reranker_score(pairs):
        return reranker_model.score_pairs(pairs, task=base.TASK)

    report, test_rows = {}, {}
    for name, score in [("reranker", reranker_score), ("reader", reader_scorer(tokenizer, reader_model))]:
        measure(panels["dev"][:1], score)
        dev = measure(panels["dev"], score)
        cuts = {arm: sorted((result[arm]["margins"][1] for _, _, result in dev), reverse=True)[cap]
                for arm in WINDOWS}
        test = measure(panels["test"], score)
        test_rows[name] = test
        report[name] = dict(thresholds=cuts, dev=summary(dev, cuts), test=summary(test, cuts))
        counts["dev"]["matched_exact_gold"] = sum(same for _, same, _ in dev)
        counts["test"]["matched_exact_gold"] = sum(same for _, same, _ in test)
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), data_sha256=base.DATA,
        code_sha256=base.digest(Path(__file__)), prior_code_sha256=base.digest(Path(base.__file__)),
        checkpoints=dict(reranker=reranker_model.provenance["revision"], reader=reader_provenance["revision"]),
        reader_minus_reranker_test=paired_model_deltas(
            test_rows["reranker"], test_rows["reader"], report["reranker"]["thresholds"], report["reader"]["thresholds"]),
        panels=counts, dev_rival_cap=cap, results=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "reranker", "reader", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
