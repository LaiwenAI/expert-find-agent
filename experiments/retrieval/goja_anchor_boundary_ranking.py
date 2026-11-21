"""Train exact experience boundaries against nearby span alternatives."""

import argparse
from functools import partial
import os
import re
from pathlib import Path

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.goja_anchor_candidate_expansion import with_compounds
from experiments.retrieval.goja_experience_binding import ARCHIVE, load_ads
from experiments.retrieval.goja_raw_experience_scope import evaluate
from experiments.retrieval.goja_scope_nomination import LINK_WORDS, TOKEN, add_lexical, gap, intersection


MAX_NEGATIVES = 4
PAIR_WEIGHT = .25


NEXT_LINK = re.compile(r"[ \t]+([^\W_]+)", re.UNICODE)
YEARS = re.compile(r"[\d.,]+[ \t]+(?:jahre?|monate?)\b", re.IGNORECASE)


def with_boundary_context(text, vocabulary):
    for span, tokens, features in with_compounds(text, vocabulary):
        preceding = text[span.start - 1] if span.start else ""
        following = text[span.end] if span.end < len(text) else ""
        prefix = text[max(0, span.start - 30):span.start]
        suffix = text[span.end:span.end + 35]
        next_word = NEXT_LINK.match(suffix)
        suffix_target = next_word is not None and next_word[1].casefold() in LINK_WORDS
        previous_words = TOKEN.findall(prefix)
        preceding_link = (bool(previous_words) and preceding.isspace() and
                          previous_words[-1].casefold() in LINK_WORDS)
        yield span, tokens, [*features, float(preceding.isalnum()), float(following.isalnum()),
                             float(suffix_target), float(preceding_link),
                             float(bool(YEARS.search(prefix))),
                             float(following in (";", ":", ",", "."))]


def boundary_pairs(spans, gold, overlap):
    pairs = []
    negatives = [i for i in range(len(spans)) if not gold[i]]
    for index in np.flatnonzero(gold):
        anchor = spans[index]

        def priority(i):
            candidate = spans[i]
            length = candidate.end - candidate.start
            shared = intersection(anchor, candidate)
            union = max(anchor.end, candidate.end) - min(anchor.start, candidate.start)
            return (-shared / union, gap(anchor, candidate),
                    abs(length - (anchor.end - anchor.start)), i)

        competing = [i for i in negatives if intersection(spans[i], anchor)]
        nearby = [i for i in negatives if not intersection(spans[i], anchor)]
        count = min(len(competing), len(nearby), MAX_NEGATIVES)
        choices = competing if overlap else nearby
        for rival in sorted(choices, key=priority)[:count]:
            pairs.append((index, rival))
    return pairs


def fit_boundary_ranker(rows, bank, *, overlap):
    design, labels, point_weights, differences, pair_weights = [], [], [], [], []
    for row in rows:
        spans, exact = row[1], row[4]
        if not spans:
            continue
        values = add_lexical(row, bank, True)
        positives, negatives = int(exact.sum()), int((~exact).sum())
        point = np.full(len(exact), 1. / len(exact))
        if positives and negatives:
            point = np.where(exact, .5 / positives, .5 / negatives)
        design.extend(values)
        labels.extend(exact)
        point_weights.extend(point)
        pairs = boundary_pairs(spans, exact, overlap)
        for positive, candidate in pairs:
            differences.append(values[positive] - values[candidate])
            pair_weights.append(PAIR_WEIGHT / len(pairs))
    if not differences:
        raise ValueError("Expected gold and rival boundary pairs")
    x, y, w = map(np.asarray, (design, labels, point_weights))
    center, scale = x.mean(axis=0), x.std(axis=0).clip(.001)
    z = (x - center) / scale
    delta = np.asarray(differences) / scale
    contrast = np.asarray(pair_weights)
    normal = z.T @ (z * w[:, None]) + delta.T @ (delta * contrast[:, None]) + np.eye(z.shape[1])
    rhs = z.T @ (y * w) + delta.T @ contrast
    weights = np.linalg.solve(normal, rhs)
    return lambda features: ((features - center) / scale) @ weights


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use Slurm and a fresh output")
    ads = list(load_ads(args.archive, minimum_links=0))
    if args.reverse:
        ads.reverse()
    ads.sort(key=lambda item: item[0])
    routes, traces = {}, {}
    traces["standard"] = {}
    routes["standard"] = evaluate(ads, trace=traces["standard"])
    limits = traces["standard"]["learned_anchor_counts"]
    traces["compound_reference"] = {}
    routes["compound_reference"] = evaluate(ads, with_compounds, trace=traces["compound_reference"],
                                             anchor_limits=limits)
    alternatives = (
        ("compound_nearby_control", with_compounds, False),
        ("compound_boundary_contrast", with_compounds, True),
        ("compound_boundary_features", with_boundary_context, None),
        ("compound_features_and_contrast", with_boundary_context, True),
    )
    for name, proposer, overlap in alternatives:
        traces[name] = {}
        fitter = ({} if overlap is None else {"anchor_fit": partial(fit_boundary_ranker, overlap=overlap)})
        routes[name] = evaluate(ads, proposer, trace=traces[name], anchor_limits=limits, **fitter)
    groups = traces["standard"]["groups"]
    if any(trace["groups"] != groups for trace in traces.values()):
        raise ValueError("The evaluation ads and anchor budgets must match")
    paired = {}
    comparisons = {
        "compound_nearby_control": ("standard", "compound_reference"),
        "compound_boundary_contrast": ("standard", "compound_reference", "compound_nearby_control"),
        "compound_boundary_features": ("standard", "compound_reference"),
        "compound_features_and_contrast": ("standard", "compound_reference", "compound_boundary_features"),
    }
    for name, references in comparisons.items():
        b = traces[name]["edge_outcomes"]["learned"]
        for reference in references:
            a = traces[reference]["edge_outcomes"]["learned"]
            paired[f"{name}_vs_{reference}"] = clustered_paired_interval(
                [int(y) - int(x) for x, y in zip(a, b, strict=True)], groups
            )
    write_json(args.output, dict(routes=routes, pair_recall_deltas=paired, source=ARCHIVE,
                                runner=digest(Path(__file__)), slurm=os.environ["SLURM_JOB_ID"],
                                reverse=args.reverse))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
