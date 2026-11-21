"""Extract annotated experience links from GOJA ads with no supplied offsets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.goja_experience_binding import ARCHIVE, Entity, load_ads
from experiments.retrieval.goja_scope_nomination import (
    BOUNDARY, LINK_WORDS, TOKEN, add_lexical, intersection, lexical_bank, lexemes,
    local_features, offsets, propose, train,
)

CUES = ("erfahr", "kennt", "vertraut", "praxis", "expertise", "beherrsch",
        "routin", "wissen", "versiert", "einarbeit", "nachweis")
CAP = 12


def group_bucket(group):
    return int(group[:8], 16) % 10


def terms(text, spans):
    return [" ".join(text[span.start:span.end].casefold().split()) for span in spans]


def anchor_features(text, span, tokens, marked, known):
    beginning = text[max(0, span.start - 20):span.start]
    following = TOKEN.findall(text[span.end:span.end + 20])
    return [len(tokens), float(marked), float(known),
            float(any(cue in tokens[-1] for cue in CUES)),
            float(any(token.isdigit() for token in tokens)),
            float(tokens[0] in LINK_WORDS), float(tokens[-1] in LINK_WORDS),
            float(not beginning.split("\n")[-1].strip(" •*-\t")),
            float(bool(following) and following[0].casefold() in LINK_WORDS)]


def anchor_candidates(text, vocabulary, *, include_cue_free=False):
    words = list(TOKEN.finditer(text))
    for i, first in enumerate(words):
        marked = False
        tokens = []
        for j in range(i, min(i + 6, len(words))):
            last = words[j]
            if j > i and BOUNDARY.search(text[words[j - 1].end():last.start()]):
                break
            tokens.append(last.group().casefold())
            marked |= any(cue in tokens[-1] for cue in CUES)
            known = " ".join(tokens) in vocabulary
            if marked or known or include_cue_free:
                span = Entity(first.start(), last.end(), "candidate")
                yield span, tuple(tokens), anchor_features(text, span, tokens, marked, known)


def feature_row(group, candidates, targets, anchor=None):
    spans, tokens, values = zip(*candidates, strict=True) if candidates else ((), (), ())
    shape = len(values[0]) if values else (9 if anchor is None else 11)
    features = np.array(values, dtype=float).reshape(-1, shape)
    lexicon = [lexemes(words) for words in tokens]
    positive = np.array([offsets(span) in targets for span in spans], dtype=bool)
    return group, spans, features, lexicon, positive, None, anchor


def fit_anchor(rows, bank):
    design, labels, weights = [], [], []
    for row in rows:
        if not row[1]:
            continue
        values, exact = add_lexical(row, bank, True), row[4]
        positives, negatives = int(exact.sum()), int((~exact).sum())
        weight = np.full(len(exact), 1. / len(exact))
        if positives and negatives:
            weight = np.where(exact, .5 / positives, .5 / negatives)
        design.extend(values)
        labels.extend(exact)
        weights.extend(weight)
    x, y, w = map(np.asarray, (design, labels, weights))
    center, scale = x.mean(axis=0), x.std(axis=0).clip(.001)
    z = (x - center) / scale
    weights = np.linalg.solve(z.T @ (z * w[:, None]) + np.eye(z.shape[1]), z.T @ (y * w))
    return lambda features: ((features - center) / scale) @ weights


def nonoverlap(spans, scores):
    choice = []
    for i in sorted(range(len(spans)), key=lambda i: (-scores[i], -(spans[i].end - spans[i].start), spans[i])):
        span = spans[i]
        if all(not intersection(span, selected) for selected, _ in choice):
            choice.append((span, float(scores[i])))
    return choice


def tune(development):
    total = sum(len(truth) for _, truth in development)
    by_score = Counter()
    correct = Counter()
    for choices, truth in development:
        for item, score in choices:
            by_score[score] += 1
            correct[score] += item in truth
    cumulative = hits = 0
    best = (0., float(np.finfo(np.float64).max))
    for score in sorted(by_score, reverse=True):
        cumulative += by_score[score]
        hits += correct[score]
        best = max(best, (2 * hits / (total + cumulative), score))
    return best[1]


def scope_row(group, text, anchor, compounds, gold=()):
    proposals = [(s, w, local_features(text, anchor, s, w)) for s, w in propose(text, anchor, compounds)]
    return feature_row(group, proposals, {offsets(span) for span in gold}, anchor)


def measures(tp, predicted, gold):
    return dict(correct=tp, predicted=predicted, gold=gold, precision=tp / max(1, predicted),
                recall=tp / max(1, gold), f1=2 * tp / max(1, predicted + gold))


def evaluate(ads, proposer=anchor_candidates, *, trace=None, anchor_limits=None, anchor_fit=fit_anchor):
    train_ads = [ad for ad in ads if group_bucket(ad[0]) < 4]
    anchor_forms = {group: Counter(terms(text, gold)) for group, text, gold in train_ads}
    scope_forms = {
        group: Counter(term for term in terms(text, set().union(*gold.values()))
                       if len(term) >= 3 and TOKEN.fullmatch(term))
        for group, text, gold in train_ads
    }
    vocab, compounds = Counter(), Counter()
    for counter in anchor_forms.values():
        vocab.update(counter)
    for counter in scope_forms.values():
        compounds.update(counter)
    anchor_training, scope_training = [], []
    for group, text, gold in train_ads:
        candidates = list(proposer(text, vocab - anchor_forms[group]))
        anchor_training.append(feature_row(group, candidates, {offsets(a) for a in gold}))
        other_terms = compounds - scope_forms[group]
        for anchor, targets in sorted(gold.items()):
            row = scope_row(group, text, anchor, other_terms, targets)
            if row[4].any() and not row[4].all():
                scope_training.append(row)
    anchor_bank = lexical_bank(anchor_training)
    anchor_model = anchor_fit(anchor_training, anchor_bank)
    scope_bank = lexical_bank(scope_training)
    scope_model = train(scope_training, [add_lexical(row, scope_bank, True) for row in scope_training])
    development, held = [], []
    for group, text, gold in ads:
        bucket = group_bucket(group)
        if bucket < 4:
            continue
        row = feature_row(group, list(proposer(text, vocab)), set())
        spans = row[1]
        scores = anchor_model(add_lexical(row, anchor_bank, False)) if spans else []
        ranked = nonoverlap(spans, scores)
        anchors = {offsets(span) for span in gold}
        if bucket < 6:
            development.append((group, text, gold, ranked, anchors))
        else:
            gazetteer = [span for span in spans if terms(text, (span,))[0] in vocab]
            held.append((group, text, gold, ranked, nonoverlap(gazetteer, [0.] * len(gazetteer)), spans))
    cutoff = tune([([(offsets(span), score) for span, score in ranked], anchors)
                   for _, _, _, ranked, anchors in development])

    def ranked_targets(group, text, anchor):
        row = scope_row(group, text, anchor, compounds)
        if not row[1]:
            return [], set()
        scores = scope_model(add_lexical(row, scope_bank, False))
        order = np.argsort(-scores, kind="stable")
        return [(offsets(row[1][i]), float(scores[i])) for i in order[:CAP]], set(map(offsets, row[1]))

    def evaluate_candidates(group, text, anchors):
        return {offsets(anchor): ranked_targets(group, text, anchor) for anchor in anchors}

    def all_truth(gold):
        return {(offsets(a), offsets(t)) for a, targets in gold.items() for t in targets}

    dev_edges, dev_candidates = [], []
    for group, text, gold, ranked, _ in development:
        anchors = [span for span, score in ranked if score >= cutoff]
        dev_edges.append(all_truth(gold))
        dev_candidates.append(evaluate_candidates(group, text, anchors))
    calibrated_edge_cutoff = tune([
        ([((a, t), score) for a, (selected, _) in candidates.items() for t, score in selected], gold)
        for candidates, gold in zip(dev_candidates, dev_edges, strict=True)
    ])
    dev_gold = sum(map(len, dev_edges))
    fixed_options = {}
    for width in range(1, 6):
        hits = issued = 0
        for candidates, gold in zip(dev_candidates, dev_edges, strict=True):
            selected = {(a, t) for a, (ranked, _) in candidates.items() for t, _ in ranked[:width]}
            issued += len(selected)
            hits += len(selected & gold)
        fixed_options[width] = measures(hits, issued, dev_gold)
    width = max(fixed_options, key=lambda n: (fixed_options[n]["f1"], -n))

    names = ("gazetteer", "learned", "learned_best_fixed", "learned_calibrated", "oracle_anchor")
    metrics = {name: Counter() for name in names}
    edge_outcomes, groups = {name: [] for name in names}, []
    true_anchors = true_edges = candidate_anchors = 0
    learned_anchor_counts = {}
    for group, text, gold, scores, exact_forms, proposed in held:
        anchor_truth = set(map(offsets, gold))
        edges = all_truth(gold)
        true_anchors += len(anchor_truth)
        true_edges += len(edges)
        candidate_anchors += len(set(map(offsets, proposed)) & anchor_truth)
        groups.extend([group] * len(edges))
        chosen = ([a for a, score in scores if score >= cutoff] if anchor_limits is None else
                  [a for a, _ in scores[:anchor_limits[group]]])
        learned_anchor_counts[group] = len(chosen)
        learned = evaluate_candidates(group, text, chosen)
        routes = (
            ("gazetteer", evaluate_candidates(group, text, [a for a, _ in exact_forms]), 1, None),
            ("learned", learned, 1, None),
            ("learned_best_fixed", learned, width, None),
            ("learned_calibrated", learned, CAP, calibrated_edge_cutoff),
            ("oracle_anchor", evaluate_candidates(group, text, gold), 1, None),
        )
        for name, candidates, count, threshold in routes:
            found = set(candidates)
            generated = {(a, t) for a, (_, pool) in candidates.items() for t in pool}
            shortlist = {(a, t) for a, (selected, _) in candidates.items() for t, _ in selected}
            issued = {(a, t) for a, (ranked, _) in candidates.items() for t, score in ranked[:count]
                      if threshold is None or score >= threshold}
            metrics[name].update(anchor_hit=len(found & anchor_truth), anchors_issued=len(found),
                                 exact_anchor_edges=sum(a in found for a, _ in edges),
                                 target_generated=len(edges & generated), target_at_12=len(edges & shortlist),
                                 best_possible_top1=len({a for a, _ in edges & shortlist}),
                                 exact_issued=len(edges & issued), pairs_issued=len(issued),
                                 ads_without_gold_with_issued=int(not gold and bool(issued)))
            edge_outcomes[name].extend(edge in issued for edge in sorted(edges))
    results = {}
    for name in names:
        m = metrics[name]
        results[name] = dict(counts=dict(m),
                             anchor=measures(m["anchor_hit"], m["anchors_issued"], true_anchors),
                             pair=measures(m["exact_issued"], m["pairs_issued"], true_edges),
                             paired_recall_vs_gazetteer=clustered_paired_interval(
                                 [int(x)-int(y) for x,y in zip(edge_outcomes[name], edge_outcomes["gazetteer"], strict=True)],
                                 groups))
    if trace is not None:
        trace.update(groups=groups, edge_outcomes=edge_outcomes, learned_anchor_counts=learned_anchor_counts)
    return dict(fit_ads=len(train_ads), calibration_ads=len(development), test_ads=len(held),
                test_anchor_proposals=sum(len(spans) for _, _, _, _, _, spans in held),
                test_ads_without_linked_experience=sum(not gold for _, _, gold, *_ in held),
                anchor_candidates_with_gold=candidate_anchors, calibration_cutoff=cutoff, calibrated_edge_cutoff=calibrated_edge_cutoff,
                selected_fixed_width=width, dev_fixed_widths=fixed_options, rows=results,
                signature=sha256(json.dumps(edge_outcomes, sort_keys=True).encode()).hexdigest())


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm allocation and a fresh output")
    ads = list(load_ads(args.archive, minimum_links=0))
    if args.reverse:
        ads.reverse()
    ads.sort(key=lambda row: row[0])
    result = evaluate(ads)
    write_json(args.output, dict(**result, source=ARCHIVE, script=digest(Path(__file__)),
                                slurm=os.environ["SLURM_JOB_ID"], reverse=args.reverse))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
