"""Compare query-directed evidence with a fixed abstract lead on held-back study clues."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import compound_contributor_search as base
from experiments.credit.factored_compound_search import ABSTRACT_SENTENCE, SOURCE_SYSTEM
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("lead", "directed")
DEPTH = 4
WORD_BUDGET = 80


def prepare(articles, fixture):
    if {doi: article.digest for doi, article in articles.items()} != fixture["sources"]:
        raise ValueError("Publisher source hashes changed")
    clues = {}
    index_articles = {}
    for doi, article in articles.items():
        sentences = ABSTRACT_SENTENCE.split(article.abstract)
        usable = [sentence for sentence in sentences[1:] if 14 <= len(sentence.split()) <= 55]
        if usable:
            held_back = usable[-1]
            clues[doi] = " ".join(word for i, word in enumerate(held_back.split()) if i % 4 != 1)
            sentences = [sentence for sentence in sentences if sentence != held_back]
        index_articles[doi] = replace(article, abstract=" ".join(sentences))
    by_target = {}
    for row in fixture["queries"]:
        if row["gold"] and row["doi"] in clues:
            doi = row["doi"]
            by_target[doi] = dict(row, clue=clues[doi])
    rows = sorted(by_target.values(), key=lambda row: base.stable("heldout-source-window-v1", row["doi"]))
    if len(rows) < 40:
        raise ValueError("At least 40 distinct target papers required")
    return index_articles, rows


def make_windows(abstract):
    words = abstract.split()
    starts = range(0, max(len(words) - WORD_BUDGET, 0) + 1, WORD_BUDGET // 2)
    starts = sorted(set(starts) | {max(len(words) - WORD_BUDGET, 0)})
    return [(" ".join(words[i:i + WORD_BUDGET]), set(base.tokens(" ".join(words[i:i + WORD_BUDGET]))))
            for i in starts]


def observe(index, windows, weights, row, state, arm, read):
    started = perf_counter()
    sources = index.sources(row, state)[:DEPTH]
    terms = set(base.tokens(row["clue"])) - base.STOP
    prompts = []
    for doi in sources:
        views = windows[doi]
        if arm == "lead":
            passage = views[0][0]
        else:
            passage = max(views, key=lambda view: sum(weights.get(term, 0) for term in sorted(terms & view[1])))[0]
        prompts.append((f"Study clue: {row['clue']}", f"Study title: {index.articles[doi].title}\nAbstract excerpt: {passage}"))
    scores, clipped = read(prompts) if prompts else ([], 0)
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid source reader scores")
    ranks = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    best = ranks[0] if ranks else None
    return dict(nominated=row["doi"] in sources, retrieved=sources[0] if sources else None,
                proposed=sources[best] if best is not None else None,
                score=float(scores[best]) if best is not None else None,
                calls=len(prompts), ms=(perf_counter() - started) * 1000)


def summarize(records, cutoff):
    counts = dict(queries=len(records), at_four=0, lexical_correct=0, lexical_wrong=0, proposed_correct=0,
                  accepted_correct=0, accepted_wrong=0, accepted_correct_owner_links=0,
                  accepted_wrong_owner_links=0, reader_pairs=0)
    for row, observation, articles in records:
        target = row["doi"]
        source = observation["proposed"]
        counts["at_four"] += observation["nominated"]
        counts["lexical_correct"] += observation["retrieved"] == target
        counts["lexical_wrong"] += observation["retrieved"] not in (None, target)
        counts["proposed_correct"] += source == target
        counts["reader_pairs"] += observation["calls"]
        if source is None or observation["score"] <= cutoff:
            continue
        correct = source == target
        required = set(row["roles"])
        owner_links = sum(required <= roles for roles in articles[source].roles)
        counts["accepted_correct" if correct else "accepted_wrong"] += 1
        counts["accepted_correct_owner_links" if correct else "accepted_wrong_owner_links"] += owner_links
    counts["p95_ms"] = float(np.quantile([item["ms"] for _, item, _ in records], .95))
    return counts


def run(fixture, checkpoint, output, folders):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    articles, rows = prepare(base.capture(folders), json.loads(fixture.read_bytes()))
    index = base.Index(articles)
    windows = {doi: make_windows(a.abstract) for doi, a in articles.items()}
    weights = {term: math.log1p((len(articles) - len(items) + .5) / (len(items) + .5))
               for term, items in index.postings.items()}
    tokenizer, model, info = base.reader_module.load_reader(checkpoint)
    read = base.reader_module.reader_scorer(tokenizer, model, system=SOURCE_SYSTEM)
    for arm in ARMS:
        observe(index, windows, weights, rows[0], "present", arm, read)
    states = ("present", "withheld")
    panels = {phase: {state: {arm: [] for arm in ARMS} for state in states}
              for phase in ("calibration", "test")}
    for i, row in enumerate(rows):
        phase = "calibration" if i < len(rows) // 2 else "test"
        for j, state in enumerate(states):
            for arm in (ARMS if (i + j) % 2 == 0 else ARMS[::-1]):
                observation = observe(index, windows, weights, row, state, arm, read)
                panels[phase][state][arm].append((row, observation, articles))
    cutoffs = {}
    for arm in ARMS:
        wrong_scores = [item["score"] for state in states for row, item, _ in panels["calibration"][state][arm]
                        if item["proposed"] not in (None, row["doi"])]
        cutoffs[arm] = max(wrong_scores)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), fixture_sha256=digest(fixture),
        code_sha256=digest(Path(__file__)), checkpoint=info["revision"], articles=len(articles), targets=len(rows),
        source_depth=DEPTH, window_word_budget=WORD_BUDGET, cutoffs=cutoffs,
        panels={phase: {state: {arm: summarize(records, cutoffs[arm]) for arm, records in panel.items()}
                        for state, panel in group.items()} for phase, group in panels.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("fixture", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    run(**vars(parser.parse_args()))
