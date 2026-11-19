"""Test one optional evidence read against human CiteEval citation edits."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_verifier import Reranker


FILES = {
    "dev.human.out": "a0cd392684239628b559e4d0255bbece0e4a9d8f31669b0c3d9a9cd2e90a1eed",
    "test.human.out": "dbf59518a2cc96522b340ae0b5ae6a032ae2d9ca2088484876f29140b7488d71",
    "dev.metric": "b861f51d0d11b19e5c11542309e401fd98fd3f4e1ca15003d4d614e7db78881f",
    "test.metric": "4dc8ee5d0de4f10916ead55152aefa8ad4b0c6497a7de7d15662ab44bffb5545",
}
ARMS = ("query_lexical", "claim_lexical", "residual_lexical", "neural_top_two", "neural_all")
TASK = "Find an additional passage containing direct evidence for the exact stated claim. Topic overlap alone is insufficient."
STOP = set("a an the it by and for from or not as at on to is in of has have that this which with".split())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(text.casefold().split())


def terms(text):
    return {w for w in tokens(text) if len(w) >= 2 and w not in STOP}


def load(directory, split):
    metric, human = [json.loads((directory / f"{split}.{suffix}").read_bytes())
                     for suffix in ("metric", "human.out")]
    seen = {rec["sample_idx"] for rec in metric}
    if len(seen) != len(metric) or seen != human.keys():
        raise ValueError("Changed response inventory")
    return metric, human


def collect(metric, human, banned):
    cases, counts = [], Counter()
    for row in metric:
        q = normalize(row["query"])
        if q in banned:
            counts["overlap_responses_removed"] += 1
            continue
        gold = human[row["sample_idx"]]
        if normalize(row["query"]) != normalize(gold["query"]) or row["prediction"] != gold["prediction"]:
            counts["different_query_or_response"] += 1
            continue
        raw = {int(p["id"]): f"{p.get('title') or ''}\n{p['text']}" for p in row["passages"]}
        contents = {i: terms(text) for i, text in raw.items()}
        frequency = Counter(w for at in contents.values() for w in at)
        idf = {w: math.log1p((len(raw) - n + .5) / (n + .5)) for w, n in frequency.items()}
        aligned = {str(s["id"]): s for s in row["prediction_sentences_and_citations"]}
        for key, annotation in gold["statements"].items():
            claim = aligned[key]
            if normalize(claim["sentence"]) != normalize(annotation["text"]):
                raise ValueError("Changed annotated statement")
            cited = set(map(int, annotation["citations"]))
            if cited != set(claim["citations"]) or not cited <= raw.keys():
                raise ValueError("Changed citation IDs")
            opinions = [edits for context, edits in zip(annotation["types"], annotation["edits"], strict=True)
                        if context == "2"]
            votes = Counter(int(x) for edits in opinions for x in set(edits.get("add-evidence", ())))
            wanted = {i for i, n in votes.items() if n >= 2}
            if not wanted <= raw.keys() - cited:
                counts["inconsistent_consensus_edits"] += 1
                continue
            if not wanted and (len(opinions) != 3 or votes):
                counts["unresolved_statements"] += 1
                continue
            unused = raw.keys() - cited
            if not unused:
                counts["no_remaining_passages"] += 1
                continue
            known = set().union(*(contents[i] for i in cited))
            cases.append(dict(q=q, query=row["query"], claim=annotation["text"], wanted=wanted, raw=raw,
                              unused=sorted(unused), contents=contents, idf=idf, known=known,
                              has_citations=bool(cited)))
            counts["positive" if wanted else "unanimous_no_add"] += 1
    return cases, dict(counts)


def lexical(query, case):
    idf = case["idf"]
    total = sum(idf.get(t, 1.) for t in sorted(query))
    return sorted(
        [(sum(idf[t] for t in sorted(query & case["contents"][j])) / total if total else 0., j)
         for j in case["unused"]], key=lambda p: (-p[0], p[1]))


def neural(reader, claim, raw, pool):
    a = perf_counter()
    margins, truncated = reader.score_pairs([(claim, raw[i]) for i in pool], task=TASK)
    chosen = max(range(len(pool)), key=lambda i: margins[i])
    return (pool[chosen], margins[chosen], len(pool), 1e3 * (perf_counter() - a), truncated)


def observe(reader, case, full_first):
    results = {}
    for method in ARMS[:3]:
        a = perf_counter()
        query = terms(case["query"] if method == "query_lexical" else case["claim"])
        if method == "residual_lexical":
            query -= case["known"]
        ranked = lexical(query, case)
        score, selected = ranked[0]
        results[method] = selected, score, 0, 1e3 * (perf_counter() - a), 0
        if method == "residual_lexical":
            shortlist = [i for _, i in ranked[:2]]
    # Cross-encoder evidence coverage and the smaller lexical shortlist use the same prompts.
    configs = (("neural_all", case["unused"]), ("neural_top_two", shortlist))
    for method, pool in configs if full_first else configs[::-1]:
        selected, score, reads, ms, trunc = neural(reader, case["claim"], case["raw"], pool)
        if method == "neural_top_two":
            ms += results["residual_lexical"][3]
        results[method] = selected, score, reads, ms, trunc
    return results, set(shortlist)


def passes(calibration, cases):
    limits = {}
    cap = math.floor(.10 * sum(not c["wanted"] for c in cases))
    for arm in ARMS:
        bad = sorted((r[0][arm][1] for r, c in zip(calibration, cases, strict=True) if not c["wanted"]), reverse=True)
        limit = bad[cap]
        limits[arm] = dict(threshold=limit, calibration_no_add_accepted=sum(x > limit for x in bad), cap=cap)
    return limits


def report(cases, observations, gates):
    results = {}
    groups = [c["q"] for c in cases]
    reference = None
    for arm in ARMS:
        scores = [row[0][arm] for row in observations]
        kept = [score[1] > gates[arm]["threshold"] for score in scores]
        output = [int(accept and s[0] in c["wanted"])
                  for accept, s, c in zip(kept, scores, cases, strict=True)]
        truth_exists = [bool(c["wanted"]) for c in cases]
        if arm == "query_lexical":
            reference = output
        strata = {}
        for name, has_citations in (("already_cited", True), ("uncited", False)):
            positives = [i for i, c in enumerate(cases) if c["has_citations"] == has_citations and c["wanted"]]
            strata[name] = dict(cases=len(positives), forced_hits=sum(scores[i][0] in cases[i]["wanted"] for i in positives),
                                gated_hits=sum(output[i] for i in positives))
        results[arm] = dict(
            forced_hits=sum(s[0] in c["wanted"] for s, c in zip(scores, cases, strict=True)),
            positive_by_existing_citations=strata,
            gated_hits=sum(output),
            gated_nonconsensus=sum(a and t and not hit for a, t, hit in zip(kept, truth_exists, output, strict=True)),
            gated_on_no_add=sum(a and not c["wanted"] for a, c in zip(kept, cases, strict=True)),
            p95_ms=float(np.quantile([s[3] for s in scores], .95)), mean_neural_passages_scored=float(np.mean([s[2] for s in scores])),
            truncated=sum(s[4] for s in scores),
            gated_hit_delta_vs_query=clustered_paired_interval(
                [a - b for a, b in zip(output, reference, strict=True)], groups),
        )
    neural_differences = {}
    for name, gated in (("forced", False), ("gated", True)):
        delta, positive_queries = [], []
        for row, case in zip(observations, cases, strict=True):
            if not case["wanted"]:
                continue
            hits = [int(row[0][arm][0] in case["wanted"]
                        and (not gated or row[0][arm][1] > gates[arm]["threshold"]))
                    for arm in ("neural_top_two", "neural_all")]
            delta.append(hits[0] - hits[1])
            positive_queries.append(case["q"])
        interval = clustered_paired_interval(delta, positive_queries)
        interval["sampling_unit"] = "normalized query"
        neural_differences[name] = interval
    return dict(
        queries=len(set(groups)),
        neural_top_two_minus_all_positive=neural_differences,
        shortlist_reaches_consensus=sum(bool(c["wanted"] & row[1]) for c, row in
                                        zip(cases, observations, strict=True)),
        arms=results,
    )


def run(data, reranker, output):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require new Slurm CUDA output")
    if any(digest(data / name) != check for name, check in FILES.items()):
        raise ValueError("Changed upstream inputs")
    dev, test = load(data, "dev"), load(data, "test")
    banned = {normalize(r["query"]) for r in dev[0]}
    calibration, dev_counts = collect(*dev, set())
    evaluation, test_counts = collect(*test, banned)
    if not calibration or not evaluation or any(c["q"] in banned for c in evaluation):
        raise ValueError("Unexpected query split")
    model = Reranker(reranker)
    observe(model, calibration[0], False)
    dev_scores = [observe(model, c, i % 2 == 0) for i, c in enumerate(calibration)]
    raw = []
    for i, c in enumerate(evaluation, 1):
        raw.append(observe(model, c, i % 2 == 0))
        if i % 200 == 0:
            print(f"{i}/{len(evaluation)} statements", flush=True)
    cutoffs = passes(dev_scores, calibration)
    summary = dict(input_sha256=FILES, script_sha256=digest(Path(__file__)), job=os.environ["SLURM_JOB_ID"],
                   gpu=torch.cuda.get_device_name(), checkpoint=model.provenance["revision"],
                   calibration=dev_counts, test=test_counts, gates=cutoffs,
                   report=report(evaluation, raw, cutoffs))
    with output.open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "reranker", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
