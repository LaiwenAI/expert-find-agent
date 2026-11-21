"""Measure candidate loss before structural binding on GOJA experience spans."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.goja_experience_binding import ARCHIVE, PAD, Entity, gap, make_cases, same_line


TOKEN = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*", re.UNICODE)
BOUNDARY = re.compile(r"[^\s-]|[\r\n]")
LINK_WORDS = {"in", "mit", "im", "beim", "bei", "von", "der", "des", "dem", "für", "zu"}
MAX_TOKENS = 5
CAPS = (1, 5, 12)


def propose(text, anchor, compounds):
    start, end = max(0, anchor.start - PAD), min(len(text), anchor.end + PAD)
    words = list(TOKEN.finditer(text, start, end))
    rows = []
    for i, first in enumerate(words):
        if (first.start() == start and start and text[start - 1].isalnum()):
            continue
        for j in range(i, min(i + MAX_TOKENS, len(words))):
            last = words[j]
            if j > i and BOUNDARY.search(text[words[j - 1].end():last.start()]):
                break
            if last.end() == end and end < len(text) and text[end].isalnum():
                break
            candidate = Entity(first.start(), last.end(), "auto")
            disjoint = candidate.end <= anchor.start or candidate.start >= anchor.end
            if disjoint:
                rows.append((candidate, tuple(w.group().casefold() for w in words[i:j + 1])))
    if compounds:
        for word in words:
            token = word.group()
            for trailing in (False, True):
                for size in range(len(token) - 1, 2, -1):
                    part = token[-size:] if trailing else token[:size]
                    if part.casefold() not in compounds:
                        continue
                    bound = (word.end() - size, word.end()) if trailing else (word.start(), word.start() + size)
                    if bound[1] <= anchor.start or bound[0] >= anchor.end:
                        rows.append((Entity(*bound, "auto"), (part.casefold(),)))
                    break
    return rows


def offsets(span):
    return span.start, span.end


def intersection(a, b):
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def lexemes(words):
    return ("phrase=" + " ".join(words), "first=" + words[0], "last=" + words[-1],
            "first_suffix=" + words[0][-3:], "last_suffix=" + words[-1][-3:])


def local_features(text, anchor, span, words):
    prefix = TOKEN.findall(text[max(0, span.start - 24):span.start])
    suffix = TOKEN.findall(text[span.end:span.end + 24])
    return [-np.log1p(gap(anchor, span)), float(same_line(text, anchor, span)),
            float(span.start >= anchor.end), len(words), np.log1p(span.end - span.start),
            float(words[0] in LINK_WORDS), float(words[-1] in LINK_WORDS),
            float(bool(prefix) and prefix[-1].casefold() in LINK_WORDS),
            float(bool(suffix) and suffix[0].casefold() in LINK_WORDS),
            float(text[span.start].isupper()), float(any(w.isdigit() for w in words))]


def training_group(group):
    return int(group[:8], 16) % 10 < 6


def prepare(source, use_compounds):
    rows = []
    cases = list(make_cases(source))
    vocabulary = Counter()
    per_ad = defaultdict(Counter)
    if use_compounds:
        for group, text, _, menu, correct in cases:
            gold = menu[correct]
            term = text[gold.start:gold.end]
            if training_group(group) and len(term) >= 3 and TOKEN.fullmatch(term):
                vocabulary[term.casefold()] += 1
                per_ad[group][term.casefold()] += 1
    for group, text, anchor, menu, correct in cases:
        gold = menu[correct]
        compounds = vocabulary - per_ad[group] if training_group(group) else vocabulary
        raw = propose(text, anchor, compounds)
        candidates = [span for span, _ in raw]
        tokens = [lexemes(words) for _, words in raw]
        features = np.array([local_features(text, anchor, span, words) for span, words in raw], dtype=np.float64)
        labels = np.array([offsets(span) == offsets(gold) for span in candidates])
        overlap = np.array([intersection(span, gold) * 2 >= max(span.end, gold.end) - min(span.start, gold.start)
                            and all(intersection(span, distractor) == 0 for distractor in menu if distractor != gold)
                            for span in candidates])
        rows.append((group, candidates, features, tokens, labels, overlap, anchor))
    return rows


def lexical_bank(rows):
    corpus, per_ad = Counter(), defaultdict(Counter)
    positive, per_positive = Counter(), defaultdict(Counter)
    for group, _, _, lex, labels, _, _ in rows:
        for i, keys in enumerate(lex):
            corpus.update(keys)
            per_ad[group].update(keys)
            if labels[i]:
                positive.update(keys)
                per_positive[group].update(keys)
    return corpus, per_ad, positive, per_positive


def add_lexical(row, bank, held):
    group, _, structure, lex, _, _, _ = row
    corpus, per_ad, positive, per_positive = bank
    own_all = per_ad[group] if held else Counter()
    own_yes = per_positive[group] if held else Counter()
    log_odds = [
        [np.log1p(positive[x] - own_yes[x]) - np.log1p(corpus[x] - own_all[x]) for x in keys]
        for keys in lex
    ]
    return np.column_stack((structure, log_odds))


def train(rows, inputs):
    design, labels, weights = [], [], []
    for row, values in zip(rows, inputs, strict=True):
        exact = row[4]
        count = int(exact.sum())
        if count:
            design.extend(values)
            labels.extend(exact)
            weights.extend(np.where(exact, .5 / count, .5 / (len(exact) - count)))
    x, y, w = map(np.asarray, (design, labels, weights))
    mean, std = x.mean(axis=0), x.std(axis=0).clip(.001)
    z = (x - mean) / std
    coef = np.linalg.solve(z.T @ (z * w[:, None]) + np.eye(z.shape[1]), z.T @ (y * w))
    return lambda data: ((data - mean) / std) @ coef


def evaluate(rows):
    train_rows = [row for row in rows if training_group(row[0])]
    test = [row for row in rows if not training_group(row[0])]
    bank = lexical_bank(train_rows)
    shape = train(train_rows, [row[2] for row in train_rows])
    lexical = train(train_rows, [add_lexical(row, bank, True) for row in train_rows])
    outcomes = defaultdict(list)
    groups = [row[0] for row in test]
    for row in test:
        _, _, features, _, exact, overlap, _ = row
        scores = {
            "nearest": features[:, 0],
            "line_first": 10 * features[:, 1] + features[:, 0],
            "trained_shape": shape(features),
            "trained_lexical": lexical(add_lexical(row, bank, False)),
        }
        for name, values in scores.items():
            ordered = np.argsort(-values, kind="stable")
            outcomes[name].append([
                *[int(exact[ordered[:k]].any()) for k in CAPS],
                *[int(overlap[ordered[:k]].any()) for k in CAPS],
            ])
    reference = np.array(outcomes["line_first"])
    measured = {}
    for arm, values in outcomes.items():
        metrics = np.array(values)
        measured[arm] = {
            metric: {str(k): int(metrics[:, i + start].sum()) for i, k in enumerate(CAPS)}
            for metric, start in (("exact", 0), ("overlap", len(CAPS)))
        }
        measured[arm]["exact_at_1_vs_line"] = clustered_paired_interval(
            (metrics[:, 0] - reference[:, 0]).tolist(), groups
        )
    signature = sha256(json.dumps(dict(outcomes), sort_keys=True).encode()).hexdigest()
    report = dict(train_anchors=len(train_rows), test_anchors=len(test), test_ads=len(set(groups)),
                  candidates=int(sum(len(row[1]) for row in test)),
                  exact_candidate_ceiling=int(sum(row[4].any() for row in test)),
                  overlap_candidate_ceiling=int(sum(row[5].any() for row in test)),
                  threshold_caps=CAPS, results=measured, signature=signature)
    return report, test, outcomes


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm allocation and a fresh results path")
    comparison = {}
    for name, use_compounds in (("standard", False), ("compound", True)):
        rows = prepare(args.archive, use_compounds)
        if args.reverse:
            rows.reverse()
        # Freeze reduction and tie ordering across discovery traversals.
        rows.sort(key=lambda row: (row[0], row[6]))
        comparison[name] = evaluate(rows)

    plain, plain_test, plain_predictions = comparison["standard"]
    compound, expanded_test, expanded_predictions = comparison["compound"]
    if [(r[0], r[6]) for r in plain_test] != [(r[0], r[6]) for r in expanded_test]:
        raise ValueError("Candidate expansion changed the held-out anchor identity")
    groups = [r[0] for r in plain_test]
    a, b = (np.array(predictions["trained_lexical"])
            for predictions in (plain_predictions, expanded_predictions))
    contrasts = {
        "exact_candidate_ceiling": clustered_paired_interval(
            [int(y[4].any()) - int(x[4].any()) for x, y in zip(plain_test, expanded_test, strict=True)], groups
        ),
    }
    for index, cap in enumerate(CAPS):
        for block, name in enumerate(("exact", "overlap")):
            column = index + block * len(CAPS)
            contrasts[f"{name}_at_{cap}"] = clustered_paired_interval(
                (b[:, column] - a[:, column]).tolist(), groups
            )

    write_json(args.output, dict(standard=plain, compound=compound, compound_vs_standard=contrasts,
                                source=ARCHIVE, script=digest(Path(__file__)),
                                slurm=os.environ["SLURM_JOB_ID"], reverse=args.reverse))



if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
