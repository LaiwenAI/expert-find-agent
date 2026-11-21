"""Evaluate same-author and author-role postings for missing-source recovery."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import nested_source_acceptance as nested
from experiments.credit.engineering_cross_source_search import Catalog, KEYWORD_STOP
from experiments.credit.engineering_contribution_panel import normalize

base, role = nested.base, nested.role
METHODS = ("global", "author", "author_role")
DISCOVERY_CAP = 12
SHALLOW, DEEP = nested.SHALLOW, nested.DEEP


class PostingIndex:
    def __init__(self, catalog):
        self.catalog = catalog
        self.position = {doc: i for i, doc in enumerate(catalog.ids)}
        self.by_person, self.by_role = defaultdict(list), defaultdict(list)
        for doc, article in sorted(catalog.articles.items()):
            for person, roles in sorted(article["roles"].items()):
                self.by_person[person].append(doc)
                for label in sorted(roles):
                    self.by_role[person, label].append(doc)

    def target(self, article, label, other_label, mode):
        members = self.catalog.articles[article]["roles"]
        neighbors, read = set(), 0
        for person, declared in members.items():
            if mode == "author":
                posting = self.by_person[person]
            elif label in declared:
                posting = self.by_role.get((person, other_label), ())
            else:
                continue
            neighbors.update(posting)
            read += len(posting)
        return neighbors, read


def source_features(topic, clues, footprints, doc, side):
    lead = topic[side]
    vocab, bigrams = footprints.index[doc]
    own = footprints.footprint(clues[side], clues[1-side])
    cross = footprints.footprint(clues[1-side], clues[side])[0]
    unigrams, rare, pairs = [nested.previous.coverage(weights, values)
                              for weights, values in zip(own, (vocab, vocab, bigrams), strict=True)]
    dense = lead["cosine"][doc]
    # Restrict features to quantities that do not depend on a paper entering
    # a global shortlist or on ranks inside the postings result.
    return np.array([dense, dense - lead["best"], dense - topic[1-side]["cosine"][doc],
                     unigrams, rare, pairs, unigrams - nested.previous.coverage(cross, vocab)])


def rrf(index, topic, terms, paper_ids):
    dense = sorted(paper_ids, key=lambda doc: (-topic["cosine"][doc], doc))
    lexical = sorted(paper_ids, key=lambda doc: (
        -sum(index.catalog.lexical[index.position[doc]].get(term, 0.) for term in terms), doc
    ))
    other = {doc: i for i, doc in enumerate(lexical)}
    return [doc for _, doc in sorted((-(1 / (61 + i) + 1 / (61 + other[doc])), doc)
                                    for i, doc in enumerate(dense))[:DISCOVERY_CAP]]


def applicable(articles, pair, labels):
    return bool(base.known_owners(articles, pair, labels))


def choices(index, topic, clues, absent, truth_requests, prediction, role_cut, footprints, training):
    articles = index.catalog.articles
    pools = [side["fused"] for side in topic]
    ordered = sorted(
        ((i+j, (a, b)) for i, a in enumerate(pools[0][:SHALLOW]) for j, b in enumerate(pools[1][:SHALLOW])
         if a != b and articles[a]["roles"].keys() & articles[b]["roles"].keys()),
    )
    shallow = ordered[0][1] if ordered else None
    cache = {}
    side_cache = [{}, {}]
    lexical = [sorted(set(normalize(clue).split()) - KEYWORD_STOP) for clue in clues]

    def features(pair):
        if pair not in cache:
            vectors = []
            for side, doc in enumerate(pair):
                if doc not in side_cache[side]:
                    side_cache[side][doc] = source_features(topic, clues, footprints, doc, side)
                vectors.append(side_cache[side][doc])
            a, b = vectors
            cache[pair] = np.r_[np.minimum(a, b), np.maximum(a, b)]
        return cache[pair]

    def make_pools(labels):
        key = labels
        if key in pool_cache:
            return pool_cache[key]
        proposals = {}
        for arm in METHODS:
            scanned, read = 0, 0
            shortlist = set()
            if arm == "global":
                scanned = sum(len(p) for p in pools)
                shortlist.update((a, b) for a, b in product(*pools) if a != b)
            else:
                for side in (0, 1):
                    anchor = pools[side][0]
                    neighbors, work = index.target(anchor, labels[side], labels[1-side], arm)
                    neighbors -= {anchor, absent}
                    scanned += len(neighbors)
                    read += work
                    for other in rrf(index, topic[1-side], lexical[1-side], neighbors):
                        shortlist.add((anchor, other) if side == 0 else (other, anchor))
            options = {pair: features(pair) for pair in sorted(shortlist) if applicable(articles, pair, labels)}
            proposals[arm] = (options, scanned, read)
        pool_cache[key] = proposals
        return proposals

    training_pool = {}
    if training:
        training_pool = {pair: features(pair) for pair in product(*pools)
                         if base.valid_pair(index.catalog, *pair)}
    pool_cache = {}
    for _, gold, phrases in truth_requests:
        labels, margin = zip(*(prediction[phrase] for phrase in phrases), strict=True)
        could_answer = min(margin) > role_cut
        baseline = shallow if shallow and could_answer and applicable(articles, shallow, labels) else None
        routes = make_pools(labels) if could_answer and baseline is None else {}
        if training:
            routes = routes | {"source_fit": (training_pool, 0, 0)}
        yield tuple(gold), labels, baseline, routes


def data(index, encoder, groups, plan, mode, predictions, role_cut, footprints, reverse, *, training=False):
    for group in sorted(groups, key=lambda row: row["papers"], reverse=reverse):
        paper = tuple(group["papers"])
        clues = [index.catalog.articles[doc]["clue"] for doc in paper]
        if mode == "keywords":
            clues = [index.catalog.keywords(clue) for clue in clues]
        embedding = encoder.encode(base.search.topic_prompts(clues))
        for absent in (None, *paper):
            ranked = index.catalog.rank(embedding, clues, absent, depth=DEEP)
            options = list(choices(index, ranked, clues, absent, plan[paper], predictions, role_cut, footprints, training))
            yield paper, absent, options


def fit(rows):
    # Each route sees a representative training row for every proposed
    # pair and availability case. Repeated role wordings do not add weight.
    seen = {}
    for paper, absent, cases in rows:
        for _, _, _, routes in cases:
            for candidates, _, _ in routes.values():
                for pair, values in candidates.items():
                    seen[paper, absent or "", pair] = values
    vectors, flags = [], []
    for (paper, _, pair), values in sorted(seen.items()):
        vectors.append(values)
        flags.append(pair == paper)
    if len(set(flags)) != 2:
        raise ValueError(f"Need true and substituted source examples, saw {Counter(flags)}")
    return base.fit(np.asarray(vectors), np.asarray(flags, dtype=float)), Counter(flags)


def scored(candidates, model):
    if not candidates:
        return ()
    values = base.estimate(model, np.array(list(candidates.values())))
    return tuple(zip(candidates, map(float, values), strict=True))


def result_is_correct(articles, papers, pair, gold, labels):
    # Sources and exact requested labels are required; membership was
    # checked before the route could issue the proof.
    return pair == papers and gold == labels and applicable(articles, papers, gold)


def calibrate(rows, model, articles):
    groups = defaultdict(dict)
    for paper, _, cases in rows:
        wrong = groups[paper]
        for gold, labels, _, routes in cases:
            for arm, (candidates, _, _) in routes.items():
                for pair, score in scored(candidates, model):
                    if not result_is_correct(articles, paper, pair, gold, labels):
                        wrong[arm] = max(score, wrong.get(arm, -1.))
    cuts, adverse = {}, {}
    for arm in METHODS:
        maxima = sorted((wrong.get(arm, -1.) for wrong in groups.values()), reverse=True)
        cuts[arm] = maxima[len(groups)//20]
        adverse[arm] = sum(value >= 0 for value in maxima)
    return cuts, adverse


def evaluate(rows, model, cuts, articles):
    totals = defaultdict(Counter)
    wrong_groups = defaultdict(set)
    for paper, absent, cases in rows:
        environment = "present" if absent is None else "missing"
        for gold, labels, initial, routes in cases:
            for method in ("shallow", *METHODS):
                metrics = totals[environment, method]
                pair, work, read, found = initial, 0, 0, 0
                if method in routes:
                    candidates, work, read = routes[method]
                    found = len(candidates)
                    pair, _ = max(
                        ((option, value) for option, value in scored(candidates, model) if value > cuts[method]),
                        key=lambda item: item[1], default=(None, None),
                    )
                    metrics["recoverable"] += paper in candidates and labels == gold
                exact = pair is not None and result_is_correct(articles, paper, pair, gold, labels)
                wrong = pair is not None and not exact
                if wrong:
                    wrong_groups[environment, method].add(paper)
                metrics.update(requests=1, strict=exact, wrong=wrong, directed=method in routes,
                               rescue_strict=exact and initial is None, rescue_wrong=wrong and initial is None,
                               source_rows=work, postings_entries=read, verifier_pairs=found)
    return {environment: {method: dict(totals[environment, method],
                                        adverse_groups=len(wrong_groups[environment, method]))
                          for method in ("shallow", *METHODS)}
            for environment in ("present", "missing")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "old_groups", "new_panel", "role_archive", "glossary",
                 "training", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and fresh output")
    archive = role.read(args.role_archive)
    old = role.read(args.old_groups)["panels"]
    panel = role.read(args.new_panel)
    if (base.digest(args.old_groups) != panel["prior_groups_sha256"]
            or archive["embedding"] != base.verify_checkpoint(args.embedding)["revision"]):
        raise ValueError("Changed frozen inputs")
    articles = base.load_catalog((args.confirmation, args.extended))
    encoder = base.QwenTorchEmbedding(args.embedding)
    index = PostingIndex(Catalog(articles, encoder))
    footprints = nested.previous.Fingerprints(index.catalog)
    definitions, fit_phrases, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, fit_phrases)
    dev, earlier, held = (base.phrase_cases(item, wording) for item in (old["calibration"], old["held"], panel))
    _, decoding = role.embed(encoder, banks, archive["training"]["weights"], (dev, earlier, held))
    role_threshold = archive["role_cutoffs"]["query_ridge_calibrated"]
    order = sorted(old["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    result = {}
    for mode in ("excerpt", "keywords"):
        def run(groups, plan, *, training=False):
            return data(index, encoder, groups, plan, mode, decoding["ridge"], role_threshold, footprints,
                        args.reverse, training=training)
        model, train_counts = fit(run(order[:18], dev, training=True))
        cuts, adverse = calibrate(run(order[18:] + old["held"]["pairs"], dev | earlier), model, articles)
        result[mode] = dict(
            counts=evaluate(run(panel["pairs"], held), model, cuts, articles),
            training_counts=dict(true=train_counts[True], wrong=train_counts[False]),
            cutoffs=cuts, adverse_calibration_groups=adverse,
        )
    base.write_json(args.output, dict(
        outcomes=result, depth=DEEP, posting_scan_rerank_cap=DISCOVERY_CAP, held_groups=len(held),
        script_sha256=base.digest(Path(__file__)), panel_sha256=base.digest(args.new_panel),
        archive_sha256=base.digest(args.role_archive),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
