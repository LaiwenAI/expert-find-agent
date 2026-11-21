"""Calibrate on deep source substitutions, then test gated author joins."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import held_source_acceptance as previous
from experiments.credit import role_first_proof_decoding as role
from experiments.credit.engineering_cross_source_search import Catalog
from expert_search.statistics import paired_interval

base = role.base
SHALLOW, DEEP = 6, 18
FIXED = ("linked_6", "linked_18", "anchored_18")
METHODS = (*FIXED, "pair_18", "source_6", "source_18", "source_nested")
FITS = ("pair", "source")


def author_join(articles, left, right, left_ranks, right_ranks):
    # First two right positions per author suffice: a proposed paper can
    # exclude at most one right position by repeating the left DOI.
    postings = defaultdict(list)
    visited = 0
    for paper in right:
        for person in articles[paper]["roles"]:
            if len(postings[person]) < 2:
                postings[person].append(paper)
            visited += 1
    best = None
    for a in left:
        for person in articles[a]["roles"]:
            visited += 1
            for b in postings.get(person, ()):
                if a != b:
                    score = left_ranks[a] + right_ranks[b], (a, b)
                    if best is None or score < best:
                        best = score
                    break
    return (None if best is None else best[1]), visited


def source_features(catalog, topic, clues, footprint):
    features = []
    for i, side in enumerate(topic):
        spans = footprint.footprint(clues[i], clues[1-i])
        others = footprint.footprint(clues[1-i], clues[i])[0]
        gaps = sorted(side["cosine"].values(), reverse=True)[:2]
        candidates = {}
        for rank, paper in enumerate(side["fused"]):
            vocab, bigrams = footprint.index[paper]
            own, rare, adjacent = [previous.coverage(terms, seen) for terms, seen in
                                   zip(spans, (vocab, vocab, bigrams), strict=True)]
            cosine = side["cosine"][paper]
            candidates[paper] = np.array([
                cosine, cosine - side["best"], gaps[0] - gaps[1], 1 / (rank + 1),
                cosine - topic[1-i]["cosine"][paper], own, rare, adjacent,
                own - previous.coverage(others, vocab),
            ])
        features.append(candidates)
    return features


def rank_groups(catalog, groups, encoder, entrance, footprint, reverse):
    ranked = {}
    for row in sorted(groups, key=lambda row: row["papers"], reverse=reverse):
        papers = tuple(row["papers"])
        clues = [catalog.articles[paper]["clue"] for paper in papers]
        if entrance == "keywords":
            clues = [catalog.keywords(text) for text in clues]
        encoded = encoder.encode(base.search.topic_prompts(clues))
        trials = []
        for absent in (None, *papers):
            topic = catalog.rank(encoded, clues, absent, depth=DEEP)
            features = source_features(catalog, topic, clues, footprint)
            options = {}
            positions = [{paper: rank for rank, paper in enumerate(side)} for side in features]
            for a, b in product(*features):
                if a == b or not catalog.articles[a]["roles"].keys() & catalog.articles[b]["roles"].keys():
                    continue
                x, y = features[0][a], features[1][b]
                options[a, b] = np.r_[np.minimum(x, y), np.maximum(x, y)]
            trials.append(dict(absent=absent, topic=topic, positions=positions, features=features, options=options))
        ranked[papers] = trials
    return ranked


def fit(ranked):
    choices, truth = defaultdict(list), defaultdict(list)
    for papers, trials in sorted(ranked.items()):
        for row in trials:
            for i, values in enumerate(row["features"]):
                for doc, vector in values.items():
                    choices["source"].append(vector)
                    truth["source"].append(doc == papers[i])
            for pair, vector in row["options"].items():
                choices["pair"].append(vector)
                truth["pair"].append(pair == papers)
    if any(set(truth[name]) != {True, False} for name in FITS):
        raise ValueError("Need both positive and negative samples")
    return {name: base.fit(np.array(choices[name]), np.array(truth[name], dtype=float)) for name in FITS}


def scores(row, models):
    parts = []
    for values in row["features"]:
        parts.append(dict(zip(values, base.estimate(models["source"], np.asarray(list(values.values()))), strict=True)))
    options = row["options"]
    paired = (dict(zip(options, base.estimate(models["pair"], np.asarray(list(options.values()))), strict=True))
              if options else {})
    return parts, paired


def calibration(ranked, models):
    maxima = defaultdict(list)
    for papers, trials in sorted(ranked.items()):
        wrong = defaultdict(list)
        for row in trials:
            source, paired = scores(row, models)
            for pair, prediction in paired.items():
                if pair != papers:
                    wrong["pair"].append(prediction)
                    wrong["source"].append(min(source[0][pair[0]], source[1][pair[1]]))
        for name in FITS:
            maxima[name].append(float(max(wrong[name], default=-1.)))
    cutoff = {name: sorted(values, reverse=True)[len(values)//20] for name, values in maxima.items()}
    return cutoff, {name: sum(value >= 0 for value in values) for name, values in maxima.items()}


def proposed(rows, articles, cutoffs, models):
    pos = rows["positions"]
    source, pair_scores = scores(rows, models)
    order = lambda pair: (pos[0][pair[0]] + pos[1][pair[1]], pair)
    in_scope = lambda pair, depth: all(pos[i][doc] < depth for i, doc in enumerate(pair))
    pairs = sorted(pair_scores, key=order)
    linked = {depth: next((pair for pair in pairs if in_scope(pair, depth)), None) for depth in (SHALLOW, DEEP)}
    anchored = next((pair for pair in pairs if any(pos[i][doc] == 0 for i, doc in enumerate(pair))
                     and all(rows["topic"][i]["best"] - rows["topic"][i]["cosine"][doc] <= .04
                             for i, doc in enumerate(pair))), None)
    paired = max((pair for pair in pairs if pair_scores[pair] > cutoffs["pair"]),
                 key=pair_scores.__getitem__, default=None)
    wave = {}
    for depth in (SHALLOW, DEEP):
        admitted = [[doc for doc, margin in side.items() if pos[i][doc] < depth and margin > cutoffs["source"]]
                    for i, side in enumerate(source)]
        found, work = author_join(articles, *admitted, *pos)
        exhaustive = next((pair for pair in pairs if in_scope(pair, depth) and
                           min(source[0][pair[0]], source[1][pair[1]]) > cutoffs["source"]), None)
        if found != exhaustive:
            raise ValueError("Author posting join disagrees with exhaustive search")
        wave[depth] = (found, work)
    route = {}
    for name, pair, depth in (("linked_6", linked[SHALLOW], SHALLOW), ("linked_18", linked[DEEP], DEEP),
                              ("anchored_18", anchored, DEEP)):
        route[name] = (pair, 0, depth * depth, False)
    route["pair_18"] = (paired, len(pairs), DEEP * DEEP, False)
    for name, depth in (("source_6", SHALLOW), ("source_18", DEEP)):
        found, work = wave[depth]
        route[name] = (found, 2 * depth, work, False)
    shallow, shallow_cost = wave[SHALLOW]
    if shallow is None:
        deep, deep_cost = wave[DEEP]
        route["source_nested"] = (deep, 2 * DEEP, shallow_cost + deep_cost, True)
    else:
        route["source_nested"] = (shallow, 2 * SHALLOW, shallow_cost, False)
    return route


def evaluate(ranked, models, cutoffs, articles, phrases, choices, role_cut):
    counts, blocks, errors = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for papers, trials in sorted(ranked.items()):
        for row in trials:
            track = "present" if row["absent"] is None else "missing"
            for arm, (pair, evaluations, join_work, expanded) in proposed(row, articles, cutoffs, models).items():
                key = track, arm
                right_sources = pair == papers
                wrong_sources = pair is not None and not right_sources
                counts[key].update(trials=1, exact_sources=right_sources, wrong_sources=wrong_sources,
                                   scores=evaluations, expansions=expanded)
                work_counter = "posting_entries" if arm.startswith("source_") else "cartesian_pairs"
                counts[key][work_counter] += join_work
                if wrong_sources:
                    errors[key].add(papers)
                for _, gold, text in phrases[papers]:
                    options = role.decode(articles, pair, choices, text)
                    valid = base.known_owners(articles, papers, gold) if right_sources else set()
                    accepted = {person for person, (margin, _) in options.items() if margin > role_cut}
                    supported = {person for person in accepted if person in valid and options[person][1] == tuple(gold)}
                    counts[key].update(requests=1, strict=bool(supported), wrong=bool(accepted - supported))
                    blocks[key][papers] += bool(supported)
    result = {track: {arm: dict(counts[track, arm], wrong_source_groups=len(errors[track, arm]))
                      for arm in METHODS} for track in ("present", "missing")}
    baseline = blocks["present", "linked_6"]
    result["strict_delta_vs_linked_6"] = {
        arm: paired_interval([blocks["present", arm][pair] - baseline[pair] for pair in sorted(ranked)])
        for arm in ("source_nested", "source_18", "anchored_18")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("confirmation", "extended", "old_groups", "new_panel", "role_archive", "glossary",
                  "training", "embedding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and fresh output")
    reference = role.read(args.role_archive)
    panel = role.read(args.new_panel)
    old = role.read(args.old_groups)["panels"]
    checkpoint = base.verify_checkpoint(args.embedding)
    if (checkpoint["revision"] != reference["embedding"] or base.digest(args.old_groups) != panel["prior_groups_sha256"]
            or tuple(panel["source_sha256"]) != base.INPUTS):
        raise ValueError("Changed source or role inputs")
    articles = base.load_catalog((args.confirmation, args.extended))
    encoder = base.QwenTorchEmbedding(args.embedding)
    catalog = Catalog(articles, encoder)
    footprint = previous.Fingerprints(catalog)
    definitions, training, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, training)
    test_plan = base.phrase_cases(panel, wording)
    old_plan = tuple(base.phrase_cases(old[name], wording) for name in ("calibration", "held"))
    _, decoding = role.embed(encoder, banks, reference["training"]["weights"], (test_plan, *old_plan))
    role_cut = reference["role_cutoffs"]["query_ridge_calibrated"]
    fit_order = sorted(old["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    reports = {}
    for entrance in ("excerpt", "keywords"):
        training = rank_groups(catalog, fit_order[:18], encoder, entrance, footprint, args.reverse)
        tuning = rank_groups(catalog, fit_order[18:] + old["held"]["pairs"], encoder, entrance, footprint, args.reverse)
        model = fit(training)
        cuts, wrong_groups = calibration(tuning, model)
        held = rank_groups(catalog, panel["pairs"], encoder, entrance, footprint, args.reverse)
        if len(held) != len(test_plan):
            raise ValueError("Missing test groups")
        reports[entrance] = dict(cutoffs=cuts, calibration_groups=len(tuning), groups_with_wrong_pair=wrong_groups,
                                 outcomes=evaluate(held, model, cuts, articles, test_plan, decoding["ridge"], role_cut))
    base.write_json(args.output, dict(
        reports=reports, shallow=SHALLOW, deep=DEEP, fit_groups=18, test_groups=len(test_plan),
        script_sha256=base.digest(Path(__file__)), panel_sha256=base.digest(args.new_panel),
        role_archive_sha256=base.digest(args.role_archive), rank_code_sha256=base.digest(
            Path(__file__).with_name("engineering_cross_source_search.py")),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
