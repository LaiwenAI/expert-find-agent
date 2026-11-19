"""Use a bounded citation review to identify misleading evidence, with human edits held out."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.citeeval import evidence_dispatch as base
from experiments.followir.core17_reranking import write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("lexical", "standalone_two", "context_two", "omit_one", "standalone_all")
KINDS = ("standalone_two", "context_two", "omit_one", "standalone_all")
MAX_CITATIONS = 4
PASSAGE_TOKENS = 170
STANDALONE = (
    "Does this cited passage contribute direct, non-misleading evidence for at least one part of the claim? "
    "Topic overlap, unsupported implications and contradictory evidence do not qualify."
)
CONTEXT = (
    "Does the target citation itself give direct, non-misleading support for any part of the claim? "
    "Use the other citation for context only. Do not credit the target with anything stated solely in the other citation."
)
OMISSION = (
    "Is the claim supported by the citation package without misleading evidence? "
    "A topically relevant citation that could mislead about the claim makes the package inadequate."
)


def collect(metric, human, banned):
    cases, counts = [], Counter()
    for row in metric:
        query = base.normalize(row["query"])
        if query in banned:
            counts["overlap_responses"] += 1
            continue
        panel = human[row["sample_idx"]]
        if base.normalize(panel["query"]) != query or panel["prediction"] != row["prediction"]:
            counts["unaligned_responses"] += 1
            continue
        passages = {int(p["id"]): f"{p.get('title') or ''}\n{p['text']}" for p in row["passages"]}
        statements = {str(s["id"]): s for s in row["prediction_sentences_and_citations"]}
        for key, item in panel["statements"].items():
            statement = statements[key]
            cited = sorted(set(map(int, item["citations"])))
            if (base.normalize(item["text"]) != base.normalize(statement["sentence"])
                    or set(statement["citations"]) != set(cited) or not set(cited) <= passages.keys()):
                raise ValueError("Unaligned statement or citation")
            if not 2 <= len(cited) <= MAX_CITATIONS:
                counts["outside_citation_count"] += 1
                continue
            contexts = [e for role, e in zip(item["types"], item["edits"], strict=True) if role == "2"]
            votes = Counter(int(i) for e in contexts for i in set(e.get("delete-misleading", ())))
            delete = {i for i, n in votes.items() if n >= 2}
            if not delete <= set(cited):
                counts["unaligned_deletions"] += 1
                continue
            if not delete and (len(contexts) != 3 or votes):
                counts["unresolved_or_nonretrieval"] += 1
                continue
            cases.append(dict(query=query, claim=item["text"], cited=cited, gold=delete,
                              passages={i: passages[i] for i in cited}))
            counts["positive" if delete else "unanimous_no_misleading_delete"] += 1
    return cases, dict(counts)


def prepare(reader, passages, cache):
    excerpts = {}
    for key, text in passages.items():
        if text not in cache:
            ids = reader.tokenizer.encode(text, add_special_tokens=False)
            cache[text] = (text if len(ids) <= PASSAGE_TOKENS else reader.tokenizer.decode(ids[:PASSAGE_TOKENS]),
                           len(ids) > PASSAGE_TOKENS)
        excerpts[key] = cache[text]
    return excerpts


def lexical(claim, excerpts):
    words = base.terms(claim)
    scores = {key: len(words & base.terms(text)) / max(1, len(words)) for key, text in excerpts.items()}
    return sorted(scores, key=lambda key: (scores[key], key)), scores


def read(reader, claim, documents, task):
    start = perf_counter()
    margins, truncated = reader.score_pairs([(claim, text) for text in documents], task=task)
    if not np.isfinite(margins).all():
        raise ValueError("Nonfinite cite-review margin")
    return margins, 1e3 * (perf_counter() - start), truncated


def observe(reader, case, cache, rotation):
    claim = case["claim"]
    pieces = prepare(reader, case["passages"], cache)
    excerpts = {key: text for key, (text, _) in pieces.items()}
    started = perf_counter()
    rank, overlap = lexical(claim, excerpts)
    old_a, old_b = rank[:2]
    lex_ms = 1e3 * (perf_counter() - started)
    results = {"lexical": (old_a, 1 - overlap[old_a], lex_ms, 0, 0)}
    order = KINDS[rotation:] + KINDS[:rotation]
    clipped = 0
    for arm in order:
        chosen = [old_a, old_b] if arm != "standalone_all" else rank
        started = perf_counter()
        if arm == "context_two":
            documents = []
            for key in chosen:
                sibling = next(other for other in reversed(rank) if other != key)
                documents.append(f"Target citation:\n{excerpts[key]}\nOther citation:\n{excerpts[sibling]}")
            task = CONTEXT
        elif arm == "omit_one":
            def package(ids):
                return "\n\n".join(f"Citation {i} · {excerpts[i]}" for i in ids)
            documents = [package(case["cited"]), package([i for i in case["cited"] if i != old_a])]
            task = OMISSION
        else:
            documents = [excerpts[key] for key in chosen]
            task = STANDALONE
        preparation_ms = 1e3 * (perf_counter() - started)
        margins, ms, truncated = read(reader, claim, documents, task)
        clipped += truncated
        if arm == "omit_one":
            selected, risk = old_a, margins[1] - margins[0]
        else:
            worst = int(np.argmin(margins))
            selected, risk = chosen[worst], -margins[worst]
        results[arm] = selected, risk, ms + preparation_ms + lex_ms, len(documents), sum(map(len, documents))
    return None if clipped else (case, results, sum(clipped for _, clipped in pieces.values()))


def cutoff(rows, arm, rate):
    wrong = sorted([result[arm][1] for case, result, _ in rows if not case["gold"]], reverse=True)
    if not wrong:
        raise ValueError("No calibration negatives")
    return wrong[math.floor(rate * len(wrong))]


def fit_cascade(rows):
    lexical_scores = [scores["lexical"][1] for _, scores, _ in rows]
    high_values = sorted({1., *map(float, np.quantile(lexical_scores, [.80, .90, .95]))})
    low_values = sorted({-1., *map(float, np.quantile(lexical_scores, [.25, .50, .75]))})
    cap = math.floor(.10 * sum(not case["gold"] for case, _, _ in rows))
    checks_cap = len(rows) // 2
    choices = []
    for high in high_values:
        for low in (value for value in low_values if value < high):
            uncertain = [row for row in rows if low < row[1]["lexical"][1] <= high]
            if len(uncertain) > checks_cap:
                continue
            values = {row[1]["standalone_two"][1] for row in uncertain}
            values.add(max(values, default=0.))
            for threshold in sorted(values):
                hits = false = wrong = 0
                for case, scores, _ in rows:
                    citation, lexical_risk, *_ = scores["lexical"]
                    accepted = lexical_risk > high
                    if low < lexical_risk <= high:
                        citation, neural_risk, *_ = scores["standalone_two"]
                        accepted = neural_risk > threshold
                    hits += int(accepted and citation in case["gold"])
                    false += int(accepted and not case["gold"])
                    wrong += int(accepted and bool(case["gold"]) and citation not in case["gold"])
                if false <= cap:
                    priority = (hits, -wrong, -false, -len(uncertain), high, low, threshold)
                    choices.append((priority, dict(high=high, low=low, neural=threshold, development_hits=hits,
                                                   development_false=false, development_calls=len(uncertain),
                                                   false_cap=cap, model_call_cap=checks_cap)))
    return max(choices, key=lambda pair: pair[0])[1]


def live_cascade(reader, rows, cache, policy):
    results = []
    for case, previous, clipped in rows:
        pieces = prepare(reader, case["passages"], cache)
        excerpts = {key: text for key, (text, _) in pieces.items()}
        start = perf_counter()
        rank, overlap = lexical(case["claim"], excerpts)
        chosen, risk = rank[0], 1. - overlap[rank[0]]
        reads = characters = 0
        accepted = risk > policy["high"]
        if policy["low"] < risk <= policy["high"]:
            pool = rank[:2]
            documents = [excerpts[i] for i in pool]
            margins, _, truncated = read(reader, case["claim"], documents, STANDALONE)
            weakest = int(np.argmin(margins))
            chosen = pool[weakest]
            accepted = not truncated and -margins[weakest] > policy["neural"]
            reads = len(pool)
            characters = sum(map(len, documents))
            if (chosen, -margins[weakest]) != previous["standalone_two"][:2]:
                raise AssertionError("Live cascade changed the standalone verdict")
        elapsed = 1e3 * (perf_counter() - start)
        combined = previous | {"cascade": (chosen, float(accepted), elapsed, reads, characters)}
        results.append((case, combined, clipped))
    return results


def report(rows, cuts):
    groups = [case["query"] for case, _, _ in rows]
    results = {}
    for arm, threshold in cuts.items():
        counts = Counter()
        times, reads, characters = [], [], []
        paired = []
        for case, result, _ in rows:
            selected, score, ms, cost, chars = result[arm]
            confirmed = int(score > threshold)
            target = case["gold"]
            hit = int(selected in target)
            if arm != "cascade":
                counts["forced_hits"] += hit
            counts["gated_hits"] += hit * confirmed
            counts["gated_wrong_citation"] += int(bool(target) and not hit and confirmed)
            counts["gated_on_no_delete"] += int(not target and confirmed)
            counts["model_calls"] += int(bool(cost))
            base_cite, base_score, *_ = result["standalone_two"]
            base_hit = int(base_cite in target and base_score > cuts["standalone_two"])
            if target:
                paired.append((case["query"], hit * confirmed - base_hit))
            times.append(ms)
            reads.append(cost)
            characters.append(chars)
        results[arm] = dict(counts, p95_ms=float(np.quantile(times,.95)), mean_reads=float(np.mean(reads)),
                            mean_document_chars=float(np.mean(characters)), hit_delta_vs_standalone_two=
                            clustered_paired_interval([delta for _, delta in paired], [group for group, _ in paired]))
    return dict(cases=len(rows), groups=len(set(groups)), positives=sum(bool(case["gold"]) for case, *_ in rows),
                clipped_input_document_occurrences=sum(row[2] for row in rows), arms=results)


def run(data, reranker, output):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output on Slurm CUDA")
    if any(base.digest(data / name) != sha for name, sha in base.FILES.items()):
        raise ValueError("Changed upstream files")
    dev, test = base.load(data, "dev"), base.load(data, "test")
    blocked = {base.normalize(row["query"]) for row in dev[0]}
    reader = Reranker(reranker)
    panels, provenance = {}, {}
    cache = {}
    for name, inputs, banned in (("development", dev, set()), ("test", test, blocked)):
        cases, counts = collect(*inputs,banned)
        observe(reader, cases[0], cache, 0)
        rows = [observe(reader, c, cache, i % len(KINDS)) for i, c in enumerate(cases)]
        panels[name] = [row for row in rows if row is not None]
        provenance[name] = dict(counts, excluded_context_cap=len(cases)-len(panels[name]))
    cuts = {arm: cutoff(panels["development"], arm, .10) for arm in ARMS}
    cascade_policy = fit_cascade(panels["development"])
    report_by_split = {}
    for name, rows in panels.items():
        evaluated = live_cascade(reader, rows, cache, cascade_policy)
        report_by_split[name] = report(evaluated, cuts | {"cascade": 0.})
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=base.digest(Path(__file__)),
        input_sha256=base.FILES, checkpoint=reader.provenance["revision"], exclusions=provenance, cuts=cuts,
        cascade_policy=cascade_policy, panels=report_by_split,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "reranker", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
