"""Compare source-pair acceptance on new author-disjoint publication groups."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import role_first_proof_decoding as role
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_cross_source_search import KEYWORD_STOP

base = role.base
METHODS = ("linked", "anchored", "paired_rank", "fixed_pair_score", "fixed_fingerprint")
CALIBRATED = METHODS[2:]


def tokens(text):
    return [word for word in normalize(text).split() if word not in KEYWORD_STOP and len(word) > 2]


def shingles(words):
    return frozenset(zip(words, words[1:]))


class Fingerprints:
    def __init__(self, catalog):
        self.idf = catalog.idf
        self.index = {}
        for paper, article in catalog.articles.items():
            words = tokens(article["indexed"])
            self.index[paper] = frozenset(words), shingles(words)

    def footprint(self, text, other):
        words, other_words = tokens(text), tokens(other)
        distinctive = set(words) - set(other_words)
        terms = {word: self.idf.get(word, 0.) for word in sorted(distinctive)}
        rare = dict(sorted(terms.items(), key=lambda entry: (-entry[1], entry[0]))[:4])
        phrases = {pair: min(self.idf.get(t, 0.) for t in pair)
                   for pair in sorted(shingles(words) - shingles(other_words))}
        return terms, rare, phrases

    def combine(self, row, clues):
        sides = []
        for i, pool in enumerate(row["pools"]):
            footprint = self.footprint(clues[i], clues[1-i])
            opposite = self.footprint(clues[1-i], clues[i])
            scores = {}
            for doc in pool:
                vocabulary, bigrams = self.index[doc]
                own = [coverage(query, candidates) for query, candidates in zip(
                    footprint, (vocabulary, vocabulary, bigrams), strict=True
                )]
                other = coverage(opposite[0], vocabulary)
                scores[doc] = np.array([*own, own[0] - other])
            leader = max(value[0] for value in scores.values())
            sides.append({doc: np.r_[value, value[0] - leader] for doc, value in scores.items()})
        return {pair: np.r_[features, np.minimum(sides[0][pair[0]], sides[1][pair[1]]),
                            np.maximum(sides[0][pair[0]], sides[1][pair[1]])]
                for pair, features in row["options"].items()}


def coverage(query, terms):
    return sum(weight for token, weight in query.items() if token in terms) / max(.001, sum(query.values()))


def rank_groups(catalog, encoder, groups, mode, footprints, reverse):
    for group in sorted(groups, key=lambda row: tuple(row["papers"]), reverse=reverse):
        pair = tuple(group["papers"])
        clues = [catalog.articles[paper]["clue"] for paper in pair]
        if mode == "keywords":
            clues = [catalog.keywords(clue) for clue in clues]
        vectors = encoder.encode(base.search.topic_prompts(clues))
        trials = []
        for withheld in (None, *pair):
            row = base.rank_candidates(catalog, vectors, clues, pair, withheld, arm="paired")
            row["fingerprint"] = footprints.combine(row, clues)
            trials.append(row)
        yield pair, trials


def fit(training, name):
    data, labels = [], []
    for papers, trials in sorted(training.items()):
        for row in trials:
            for pair, vector in row[name].items():
                data.append(vector)
                labels.append(pair == papers)
    if len(set(labels)) != 2:
        raise ValueError("Need present and substituted source options in fitting data")
    return base.fit(np.asarray(data), np.asarray(labels, dtype=float))


def dispatch(row, models):
    options = base.dispatch(row, (None, models["options"]), ("linked", "anchored", "paired"))
    linked, _ = options["linked"]
    decisions = dict(linked=options["linked"], anchored=options["anchored"], paired_rank=options["paired"])
    for name, features in (("fixed_pair_score", "options"), ("fixed_fingerprint", "fingerprint")):
        decision = (linked, float(base.estimate(models[features], row[features][linked][None, :])[0])) if linked else (None, None)
        decisions[name] = decision
    return decisions


def calibration(trials, models):
    by_group = {method: [] for method in CALIBRATED}
    for papers, rows in sorted(trials.items()):
        wrong = {method: [] for method in CALIBRATED}
        for row in rows:
            picks = dispatch(row, models)
            for method in CALIBRATED:
                pair, value = picks[method]
                if pair is not None and pair != papers:
                    wrong[method].append(float(value))
        for method in CALIBRATED:
            by_group[method].append(max(wrong[method], default=-100.))
    return {policy: {method: sorted(scores, reverse=True)[index]
                     for method, scores in by_group.items()}
            for policy, index in (("zero", 0), ("up_to_five_percent", len(trials)//20))}


def evaluate(trials, models, cutoffs, articles, requests, decoded, role_cut):
    counts, adverse, source_errors = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for papers, rows in sorted(trials.items()):
        for row in rows:
            track = "present" if row["withheld"] is None else "missing"
            for method, (pair, value) in dispatch(row, models).items():
                for policy, cuts in cutoffs.items():
                    if method not in CALIBRATED and policy != "zero":
                        continue
                    key = track, method, policy
                    proposed = pair if pair is not None and value > cuts.get(method, 0.) else None
                    good_source = proposed == papers
                    measurements = counts[key]
                    measurements.update(trials=1, reachable=papers in row["options"],
                                        exact_sources=good_source, wrong_sources=proposed is not None and not good_source)
                    if proposed is not None and not good_source:
                        source_errors[key].add(papers)
                    for _, gold, texts in requests[papers]:
                        reference = base.known_owners(articles, papers, gold) if track == "present" else set()
                        actual = {owner: labels for owner, (margin, labels) in role.decode(
                            articles, proposed, decoded, texts).items() if margin > role_cut}
                        exact = {owner for owner, labels in actual.items()
                                 if good_source and owner in reference and labels == tuple(gold)}
                        wrong = bool(actual.keys() - exact)
                        measurements.update(requests=1, positives=bool(reference), exact_proofs=bool(exact),
                                            wrong_proofs=wrong, wrong_source_proofs=wrong and not good_source)
                        if wrong:
                            adverse[key].add(papers)
    result = defaultdict(dict)
    for (track, method, policy), values in counts.items():
        label = method if method not in CALIBRATED else f"{method}_{policy}"
        key = track, method, policy
        result[track][label] = dict(values, adverse_groups=len(adverse[key]),
                                    wrong_source_groups=len(source_errors[key]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "prior", "new_panel", "role_archive", "niso",
                 "training", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and fresh output")
    panel = role.read(args.new_panel)
    prior = role.read(args.prior)["panels"]
    reference = role.read(args.role_archive)
    info = base.verify_checkpoint(args.embedding)
    if (reference["embedding"] != info["revision"] or panel["prior_groups_sha256"] != base.digest(args.prior)
            or tuple(panel["source_sha256"]) != base.INPUTS):
        raise ValueError("Changed role archive or frozen source inputs")
    articles = base.load_catalog((args.confirmation, args.extended))
    encoder = base.QwenTorchEmbedding(args.embedding)
    catalog = base.search.Catalog(articles, encoder)
    footprints = Fingerprints(catalog)
    definitions, training, wording = base.queries(args.niso, args.training)
    banks, _ = role.train(encoder, definitions, training)
    plan = base.phrase_cases(panel, wording)
    existing = [base.phrase_cases(prior[name], wording) for name in ("calibration", "held")]
    _, decoded = role.embed(encoder, banks, reference["training"]["weights"], (plan, *existing))
    role_cut = reference["role_cutoffs"]["query_ridge_calibrated"]
    fit_order = sorted(prior["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    fit_groups = fit_order[:18]
    tune_groups = fit_order[18:] + prior["held"]["pairs"]
    results = {}
    for mode in ("excerpt", "keywords"):
        training = dict(rank_groups(catalog, encoder, fit_groups, mode, footprints, args.reverse))
        tuning = dict(rank_groups(catalog, encoder, tune_groups, mode, footprints, args.reverse))
        models = {name: fit(training, name) for name in ("options", "fingerprint")}
        cut = calibration(tuning, models)
        held = dict(rank_groups(catalog, encoder, panel["pairs"], mode, footprints, args.reverse))
        results[mode] = dict(held=evaluate(held, models, cut, articles, plan, decoded["ridge"], role_cut), cutoffs=cut)
    base.write_json(args.output, dict(
        results=results, fit_groups=len(fit_groups), tune_groups=len(tune_groups), held_groups=len(panel["pairs"]),
        catalog_articles=len(articles), panel_sha256=base.digest(args.new_panel), embedding=info["revision"],
        role_archive_sha256=base.digest(args.role_archive), script_sha256=base.digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
