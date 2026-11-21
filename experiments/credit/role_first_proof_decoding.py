"""Separate requested-role decoding from author membership at fixed source proposals."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import held_open_proof_search as base
from experiments.followir.core17_reranking import digest, write_json

BANKS = ("ridge", "prototype", "definition")
DECODERS = ("ridge", "mixture")
CONTROLS = ("author_conditioned", "strict_author_conditioned", "bank_agreement")
ARMS = (*CONTROLS, *(f"query_{name}_{policy}" for name in DECODERS for policy in ("all", "calibrated")))


def read(path):
    return json.loads(path.read_bytes())


def normalize_scores(values):
    return (values - values.mean(axis=-1, keepdims=True)) / np.maximum(
        .01, values.std(axis=-1, keepdims=True)
    )


def fit_banks(vectors, dictionary):
    roles, count, width = vectors.shape
    flat = vectors.reshape(roles * count, width)
    prototypes = vectors.mean(axis=1)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    targets = np.repeat(np.eye(roles, dtype=np.float32), count, axis=0)
    kernel = flat @ flat.T
    kernel.flat[::len(flat) + 1] += .1
    ridge = np.linalg.solve(kernel, targets).T @ flat
    return dict(ridge=ridge, prototype=prototypes, definition=dictionary)


def bank_scores(vectors, banks):
    return np.stack([normalize_scores(vectors @ banks[bank].T) for bank in BANKS], axis=0)


def train(encoder, definitions, phrases):
    dictionary = encoder.encode(definitions)
    phrases = base.encode(encoder, phrases).reshape(len(base.ROLES), 6, -1)
    oof_scores = []
    for column in range(phrases.shape[1]):
        training_columns = np.arange(phrases.shape[1]) != column
        banks = fit_banks(phrases[:, training_columns], dictionary)
        oof_scores.append(bank_scores(phrases[:, column], banks))
    oof_scores = np.concatenate(oof_scores, axis=1)
    labels = np.tile(np.arange(len(base.ROLES)), phrases.shape[1])
    grid = [weights for weights in product(range(5), repeat=3) if sum(weights) == 4]

    def score(weights):
        logits = np.einsum("b,bqr->qr", np.asarray(weights) / 4, oof_scores)
        return int((logits.argmax(axis=1) == labels).sum())

    # Equal training accuracy favors fewer active banks, then more ridge weight.
    weights = max(grid, key=lambda w: (score(w), -np.count_nonzero(w), w))
    summary = dict(bank_order=BANKS, weights=(np.asarray(weights) / 4).tolist(),
                   oof_mixture_exact=score(weights), oof_ridge_exact=score((4, 0, 0)),
                   oof_phrases=len(labels))
    return fit_banks(phrases, dictionary), summary


def embed(encoder, banks, weights, plans):
    phrases = sorted({phrase for plan in plans for cases in plan.values()
                      for _, _, texts in cases for phrase in texts})
    vectors = base.encode(encoder, phrases)
    raw = dict(zip(phrases, zip(vectors @ banks["ridge"].T, vectors @ banks["definition"].T, strict=True),
                   strict=True))
    normalized = bank_scores(vectors, banks)
    mixture = np.einsum("b,bqr->qr", weights, normalized)
    choices = {}
    for name, matrix in zip(DECODERS, (normalized[0], mixture), strict=True):
        ranked = np.argsort(-matrix, axis=1, kind="stable")
        top = ranked[:, 0]
        margin = matrix[np.arange(len(phrases)), top] - matrix[np.arange(len(phrases)), ranked[:, 1]]
        choices[name] = {phrase: (base.ROLES[index], float(gap))
                         for phrase, index, gap in zip(phrases, top, margin, strict=True)}
    return raw, choices


def decode(articles, pair, choices, phrases):
    if pair is None:
        return {}
    labels, margins = zip(*(choices[text] for text in phrases), strict=True)
    return {owner: (min(margins), labels) for owner in base.known_owners(articles, pair, labels)}


def cutoffs(articles, plan, raw, choices, original_cut):
    cut = {
        "author_conditioned": original_cut,
        "strict_author_conditioned": base.calibrate_roles(articles, plan, raw, False),
        "bank_agreement": base.calibrate_roles(articles, plan, raw, True),
    }
    for kind, choice in choices.items():
        worst = []
        for pair, cases in plan.items():
            failures = []
            for _, roles, phrases in cases:
                reference = base.known_owners(articles, pair, roles)
                failures.extend(score for owner, (score, labels) in decode(articles, pair, choice, phrases).items()
                                if owner not in reference or labels != tuple(roles))
            worst.append(max(failures, default=-1.))
        cut[f"query_{kind}_calibrated"] = sorted(worst, reverse=True)[len(worst) // 20]
        cut[f"query_{kind}_all"] = -1.
    return cut


def source_proposals(catalog, encoder, plan, mode, reverse):
    for papers in sorted(plan, reverse=reverse):
        clues = [catalog.articles[p]["clue"] for p in papers]
        if mode == "keywords":
            clues = [catalog.keywords(text) for text in clues]
        vectors = encoder.encode(base.search.topic_prompts(clues))
        for withheld in (None, *papers):
            ranks = base.rank_candidates(catalog, vectors, clues, papers, withheld, arm="linked")
            found, _ = base.dispatch(ranks, None, ("linked",))["linked"]
            yield papers, withheld, found


def audit_failure(metrics, reference, found, papers, answer, gold, predicted):
    if not reference:
        return
    if found != papers:
        metrics["source_not_found"] += 1
    elif not answer:
        metrics["no_proof_on_correct_sources"] += 1
    elif not (answer.keys() & reference):
        metrics["wrong_identity_only"] += 1
    elif not any(answer[owner] == tuple(gold) for owner in answer.keys() & reference):
        metrics["correct_identity_wrong_roles"] += 1
        for name, labels in predicted.items():
            metrics[f"global_{name}_both_true_with_wrong_rendered_role"] += labels == tuple(gold)


def quality(articles, plan, proposals, raw, choices, cut):
    counts, groups = defaultdict(Counter), defaultdict(set)
    diagnostic = Counter()
    for papers, withheld, pair in proposals:
        track = "present" if withheld is None else "missing"
        for variant, roles, phrases in plan[papers]:
            reference = base.known_owners(articles, papers, roles) if withheld is None else set()
            scores = [raw[phrase] for phrase in phrases]
            original = base.candidates_with_roles(articles, pair, scores)
            options = dict.fromkeys(CONTROLS[:2], original)
            options["bank_agreement"] = base.candidates_with_roles(articles, pair, scores, True)
            for kind, choice in choices.items():
                decoded = decode(articles, pair, choice, phrases)
                options[f"query_{kind}_all"] = options[f"query_{kind}_calibrated"] = decoded
            for arm in ARMS:
                actual = {owner: labels for owner, (score, labels) in options[arm].items() if score > cut[arm]}
                correct = {owner for owner, labels in actual.items() if owner in reference and labels == tuple(roles)}
                wrong = bool(actual) and (pair != papers or actual.keys() != correct)
                counts[track, arm].update(
                    requests=1, positives=bool(reference), issued=bool(actual),
                    strict=pair == papers and bool(correct), wrong=wrong,
                    wrong_negative=wrong and (variant != "positive" or withheld is not None),
                    owner_correct=pair == papers and bool(actual.keys() & reference),
                )
                if wrong:
                    groups[track, arm].add(papers)
                if arm == "author_conditioned" and track == "present":
                    predicted = {name: tuple(choice[phrase][0] for phrase in phrases)
                                 for name, choice in choices.items()}
                    audit_failure(diagnostic, reference, pair, papers, actual, roles, predicted)
    return dict(
        counts={track: {arm: dict(counts[track, arm], adverse_groups=len(groups[track, arm]))
                        for arm in ARMS} for track in ("present", "missing") if counts[track, ARMS[0]]},
        baseline_positive_failures=dict(diagnostic),
    )


def recognition(plan, choices):
    gold = {text: role for cases in plan.values() for _, roles, phrases in cases
            for role, text in zip(roles, phrases, strict=True)}
    return {name: dict(exact=sum(choice[text][0] == role for text, role in gold.items()), distinct=len(gold))
            for name, choice in choices.items()}


def external(manifest, notes, wording):
    articles = {paper: dict(roles={person: set(row["roles"]) for person, row in people.items()})
                for paper, people in notes.items()}
    plan = {}
    labels = defaultdict(list)
    for index, _, phrase in wording["niso"]:
        labels[base.ROLES[index]].append(phrase)
    seen = set()
    for row in manifest["panels"]["plos"]["pairs"]:
        pair = tuple(row["papers"])
        people = articles[pair[0]]["roles"].keys() | articles[pair[1]]["roles"].keys()
        if people & seen or pair in plan:
            raise ValueError("PLOS groups overlap")
        seen.update(people)
        cases = []
        contrasts = row.get("contrasts") or [(roles, roles[::-1]) for roles in row["bindings"]]
        for index, contrast in enumerate(contrasts):
            family = base.stable(pair, index)
            for variant, roles in zip(("positive", "contrast"), contrast, strict=True):
                if bool(base.known_owners(articles, pair, roles)) != (variant == "positive"):
                    raise ValueError("PLOS declarations do not satisfy frozen contrast")
                phrases = [min(labels[role], key=lambda phrase: base.stable(family, role, phrase)) for role in roles]
                cases.append((variant, roles, phrases))
        plan[pair] = cases
    return articles, plan


def check_baseline(results, reference):
    for mode, report in results.items():
        for track, counts in report["counts"].items():
            for arm, old in zip(CONTROLS, ("role_index", "tight_role_index", "bank_agreement"), strict=True):
                expected = reference["results"][mode]["role_ablation"][track][old]
                for key, source in (("strict", "proof_correct"), ("wrong", "proof_wrong"),
                                    ("owner_correct", "owner_correct"), ("requests", "requests")):
                    if counts[arm][key] != expected[source]:
                        raise ValueError(f"Archived control changed: {mode} {track} {arm} {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "frozen", "open_archive", "glossary", "training", "embedding",
                 "plos_swaps", "plos_swap_notes", "plos_edits", "plos_edit_notes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and fresh output")
    archive = read(args.open_archive)
    info = base.verify_checkpoint(args.embedding)
    if archive["group_sha256"] != digest(args.frozen) or archive["embedding"] != info["revision"]:
        raise ValueError("Changed baseline group manifest or model")
    frozen = read(args.frozen)["panels"]
    articles = base.load_catalog((args.confirmation, args.extended))
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, training, wording = base.queries(args.glossary, args.training)
    banks, training_summary = train(encoder, definitions, training)
    catalog = base.search.Catalog(articles, encoder)
    held, dev = (base.phrase_cases(frozen[panel], wording) for panel in ("held", "calibration"))
    raw, choices = embed(encoder, banks, training_summary["weights"], (held, dev))
    cuts = cutoffs(articles, dev, raw, choices, archive["role_cutoff"])
    dev_result = quality(articles, dev, ((pair, None, pair) for pair in dev), raw, choices, cuts)
    elife = {mode: quality(articles, held, source_proposals(catalog, encoder, held, mode, args.reverse),
                           raw, choices, cuts) for mode in ("excerpt", "keywords")}
    check_baseline(elife, archive)
    plos = {}
    for name, manifest, notes in (("source_swap", args.plos_swaps, args.plos_swap_notes),
                                  ("role_edit", args.plos_edits, args.plos_edit_notes)):
        plos_articles, plan = external(read(manifest), read(notes), wording)
        external_raw, external_choices = embed(encoder, banks, training_summary["weights"], (plan,))
        plos[name] = quality(plos_articles, plan, ((pair, None, pair) for pair in sorted(plan, reverse=args.reverse)),
                             external_raw, external_choices, cuts)
        plos[name]["role_recognition"] = recognition(plan, external_choices)
        plos[name]["input_sha256"] = dict(manifest=digest(manifest), notes=digest(notes))
    write_json(args.output, dict(
        catalog_articles=len(articles), embedding=info["revision"], training=training_summary,
        elife=elife, elife_phrase_recognition=recognition(held, choices), plos=plos, role_cutoffs=cuts,
        calibration=dev_result, baseline_sha256=digest(args.open_archive), frozen_sha256=digest(args.frozen),
        script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
