"""Isolate locating versus selecting one missing QASPER evidence paragraph."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.followir.core17_verifier import Reranker
from experiments.qasper.adaptive_evidence import evidence_sets, paper_inputs, read_data
from experiments.qasper.evidence_gap_audit import choose
from experiments.qasper.evidence_shortlist import terms


ARMS = ("question_full", "anchored_full", "question_top4", "anchored_top4",
        "gap_top4", "gap_top8")
TASK = "Find a further paragraph that provides evidence needed to answer the question."


def holdout(pid, refs):
    """Reveal all but one paragraph of a reference, excluding already complete views."""
    choices = {}
    for ref in refs:
        for hidden in ref:
            shown = ref - {hidden}
            if not shown or shown in choices or any(alt <= shown for alt in refs):
                continue
            choices[shown] = {next(iter(alt - shown)) for alt in refs if len(alt - shown) == 1}
    if not choices:
        return None

    def key(shown):
        return sha256(f"qasper-completion-v1:{pid}:{sorted(shown)}".encode()).digest()

    shown = min(choices, key=key)
    return sorted(shown), choices[shown]


def shortlists(question, stats, shown, *, include_rankings=False):
    query = set(terms(question))
    remainder = query.difference(*(stats[idx] for idx in shown))
    frequency = Counter(token for doc in stats for token in doc)
    weights = {word: math.log1p((len(stats) - frequency[word] + .5) / (frequency[word] + .5))
               for word in query}
    shown_set = set(shown)
    candidates = [i for i in range(len(stats)) if i not in shown_set]
    hits = [stats[i] & query for i in range(len(stats))]
    raw = [sum(weights[w] for w in sorted(hit)) for hit in hits]
    gap = [sum(weights[w] for w in sorted(hit & remainder)) for hit in hits]
    by_question = sorted(candidates, key=lambda i: (-raw[i], i))
    by_gap = sorted(candidates, key=lambda i: (-(raw[i] + gap[i]), i))
    pools = {
        "question_full": candidates, "anchored_full": candidates,
        "question_top4": by_question[:4], "anchored_top4": by_question[:4],
        "gap_top4": by_gap[:4], "gap_top8": by_gap[:8],
    }
    if include_rankings:
        pools.update(question_ranked=by_question, gap_ranked=by_gap)
    return pools


def excerpt(tokenizer, text, cap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= cap else tokenizer.decode(ids[:cap], skip_special_tokens=True)


def observe(reader, pid, qa, paper, rng):
    paragraphs, sections = paper_inputs(paper)
    ids = {paragraph: idx for idx, paragraph in enumerate(paragraphs)}
    refs = {frozenset(ids[p] for p in ref) for ref in evidence_sets(qa, paragraphs)}
    view = holdout(pid, refs)
    if view is None:
        return None
    shown, witnesses = view
    question = qa["question"]
    stats = [set(terms(text)) for text in paragraphs]
    pools = shortlists(question, stats, shown)
    chunks = [f"{sections[i]}\n{paragraphs[i]}" for i in range(len(paragraphs))]
    snippets = [excerpt(reader.tokenizer, paragraphs[i], 128) for i in shown]
    anchored = question + "\nEvidence already available:\n" + "\n".join(snippets)

    result = {}
    for arm in rng.sample(ARMS, len(ARMS)):
        pool = pools[arm]
        query = question if arm.startswith("question_") else anchored
        begin = perf_counter()
        margins = []
        truncated = 0
        for offset in range(0, len(pool), 16):
            batch = [(query, chunks[i]) for i in pool[offset:offset + 16]]
            scores, clipped = reader.score_pairs(batch, task=TASK)
            margins.extend(scores)
            truncated += clipped
        best = max(zip(margins, pool, strict=True), key=lambda row: (row[0], -row[1]))[1]
        result[arm] = dict(hit=best in witnesses, reachable=bool(witnesses.intersection(pool)),
                           pairs=len(pool), clipped=truncated, ms=(perf_counter() - begin) * 1000)
    return result


def summarize(rows):
    summary = {}
    for arm in ARMS:
        observations = [row[arm] for row in rows]
        hits = [obs["hit"] for obs in observations]
        delta = [int(hit) - int(row["anchored_full"]["hit"]) for hit, row in zip(hits, rows, strict=True)]
        reachable = sum(obs["reachable"] for obs in observations)
        summary[arm] = dict(
            hit=sum(hits), reachable=reachable, shortlist_misses=len(rows) - reachable,
            selection_misses=reachable - sum(hits), gains=delta.count(1), losses=delta.count(-1),
            difference_ci=paired_interval(delta), mean_pairs=mean(obs["pairs"] for obs in observations),
            truncations=sum(obs["clipped"] for obs in observations),
            rerank_p95_ms=float(np.quantile([obs["ms"] for obs in observations], .95)),
        )
    return summary


def run(data, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    papers = read_data(data, "test")
    selected = choose(papers, multi=True)
    reader = Reranker(checkpoint)
    if reader.provenance["repository"] != "Qwen/Qwen3-Reranker-0.6B":
        raise ValueError("Expected the frozen Qwen3 reranker")
    reader.score_pairs([("Warm up", "A passage")] * 4, task=TASK)
    rng = random.Random(467)
    rows = []
    for idx, (pid, qa) in enumerate(selected, start=1):
        observed = observe(reader, pid, qa, papers[pid], rng)
        if observed is not None:
            rows.append(observed)
        if idx % 25 == 0:
            print(f"{idx}/{len(selected)}", flush=True)
    result = dict(dataset="QASPER v0.3 official test", selected=len(selected), evaluated=len(rows),
                  skipped=len(selected) - len(rows), gpu=torch.cuda.get_device_name(),
                  versions={"torch": torch.__version__, "cuda": torch.version.cuda},
                  checkpoint=reader.provenance, arms=summarize(rows))
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
