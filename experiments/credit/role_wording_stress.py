"""Test frozen contributor-role decoders on separate wording families."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import role_first_proof_decoding as prior
from experiments.credit.engineering_contribution_panel import normalize

base = prior.base
SCALES = (1., 4., 16.)
VIEWS = ("terse", "contrast")
ARMS = ("strict_author", "agreement", "ridge", "prototype", "kernel")


def fit_kernel(vectors, scale):
    labels = np.repeat(np.eye(len(base.ROLES), dtype=np.float32), vectors.shape[1], axis=0)
    flat = vectors.reshape(len(labels), -1)
    kernel = np.exp(scale * (flat @ flat.T - 1))
    kernel.flat[::len(kernel) + 1] += .1
    return flat, np.linalg.solve(kernel, labels)


def kernel_scores(vectors, model, scale):
    samples, weights = model
    return np.exp(scale * (vectors @ samples.T - 1)) @ weights


def select_kernel(vectors):
    count = vectors.shape[1]
    folds = []
    for i in range(count):
        folds.append((vectors[:, np.arange(count) != i], vectors[:, i]))
    gold = np.arange(len(base.ROLES))
    correct = {scale: sum(int((kernel_scores(hold, fit_kernel(fit, scale), scale).argmax(axis=1) == gold).sum())
                          for fit, hold in folds) for scale in SCALES}
    scale = max(SCALES, key=lambda candidate: (correct[candidate], -candidate))
    return scale, fit_kernel(vectors, scale), correct


def choices(vectors, phrases, banks, kernel, scale):
    ridge = vectors @ banks["ridge"].T
    definition = vectors @ banks["definition"].T
    raw = dict(zip(phrases, zip(ridge, definition, strict=True), strict=True))
    scores = {"ridge": ridge, "prototype": vectors @ banks["prototype"].T,
              "kernel": kernel_scores(vectors, kernel, scale)}
    decisions = {}
    for name, values in scores.items():
        values = prior.normalize_scores(values)
        first, second = np.argsort(-values, axis=1, kind="stable")[:, :2].T
        rows = np.arange(len(values))
        decisions[name] = {phrase: (base.ROLES[i], float(gap)) for phrase, i, gap in
                           zip(phrases, first, values[rows, first] - values[rows, second], strict=True)}
    return raw, decisions


def fit_cut(articles, plan, options):
    worst = []
    for pair, cases in plan.items():
        failures = []
        for _, roles, phrases in cases:
            reference = base.known_owners(articles, pair, roles)
            failures.extend(score for owner, (score, labels) in prior.decode(articles, pair, options, phrases).items()
                            if owner not in reference or labels != tuple(roles))
        worst.append(max(failures, default=-1.))
    return sorted(worst, reverse=True)[len(worst) // 20]


def stress_sets(path, training, wording, original):
    bank = prior.read(path)
    if set(bank) != set(base.ROLES) or any(set(rows) != set(VIEWS) for rows in bank.values()):
        raise ValueError("Expected every taxonomy role in its two wording styles")
    prior_text = {normalize(phrase) for phrase in training}
    prior_text.update(normalize(phrase) for rows in wording.values() for _, _, phrase in rows)
    seen = set()
    for rows in bank.values():
        for view, phrases in rows.items():
            if len(phrases) != 2:
                raise ValueError(f"Expected two paraphrases per role in {view}")
            for phrase in phrases:
                key = normalize(phrase)
                if key in prior_text or key in seen:
                    raise ValueError(f"Repeated wording: {phrase}")
                seen.add(key)
    plans = {}
    for view in VIEWS:
        for lane in range(2):
            plans[f"{view}_{lane}"] = {
                pair: [(variant, roles, [bank[role][view][lane] for role in roles]) for variant, roles, _ in cases]
                for pair, cases in original.items()
            }
    return bank, plans


def recognition(bank, decoded):
    scores = {}
    for view in VIEWS:
        scores[view] = {}
        for name, outputs in decoded.items():
            wrong = defaultdict(Counter)
            exact = 0
            for role, wording in bank.items():
                for phrase in wording[view]:
                    got, _ = outputs[phrase]
                    exact += got == role
                    if got != role:
                        wrong[role][got] += 1
            scores[view][name] = dict(correct=exact, phrases=2 * len(bank), confusion=wrong)
    return scores


def quality(articles, plan, proposals, raw, decoded, cuts):
    metrics, adverse = defaultdict(Counter), defaultdict(set)
    for papers, withheld, pair in proposals:
        track = "present" if withheld is None else "missing"
        for _, gold, phrases in plan[papers]:
            reference = base.known_owners(articles, papers, gold) if not withheld else set()
            author_scores = [raw[phrase] for phrase in phrases]
            answers = {
                "strict_author": base.candidates_with_roles(articles, pair, author_scores),
                "agreement": base.candidates_with_roles(articles, pair, author_scores, True),
            }
            answers.update({name: prior.decode(articles, pair, decoded[name], phrases) for name in decoded})
            for arm, options in answers.items():
                supported = {person for person, (_, cited) in options.items()
                             if pair == papers and person in reference and cited == tuple(gold)}
                accepted = {person for person, (confidence, _) in options.items() if confidence > cuts[arm]}
                wrong = bool(accepted - supported)
                metrics[track, arm].update(
                    requests=1, positive=bool(reference), strict=bool(accepted & supported),
                    wrong=wrong, issued=bool(accepted),
                    strict_before_cut=bool(supported), wrong_before_cut=bool(options.keys() - supported),
                )
                if wrong:
                    adverse[track, arm].add(papers)
    return {track: {arm: dict(metrics[track, arm], adverse_groups=len(adverse[track, arm])) for arm in ARMS}
            for track in ("present", "missing")}


def phrase_list(plans):
    return sorted({phrase for plan in plans for cases in plan.values() for _, _, texts in cases for phrase in texts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "frozen", "reference", "niso", "training",
                 "wording", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and a fresh output")
    archive = prior.read(args.reference)
    info = base.verify_checkpoint(args.embedding)
    if archive["frozen_sha256"] != base.digest(args.frozen) or archive["embedding"] != info["revision"]:
        raise ValueError("Changed role-first archive inputs")
    frozen = prior.read(args.frozen)["panels"]
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, training, wording = base.queries(args.niso, args.training)
    training_vectors = base.encode(encoder, training).reshape(len(base.ROLES), 6, -1)
    dictionary = encoder.encode(definitions)
    banks = prior.fit_banks(training_vectors, dictionary)
    if archive["training"]["weights"] != [0., 1., 0.]:
        raise ValueError("Frozen comparison expected a prototype-only winning bank")
    scale, kernel, selection = select_kernel(training_vectors)
    articles = base.load_catalog((args.confirmation, args.extended))
    catalog = base.search.Catalog(articles, encoder)
    original, dev = (base.phrase_cases(frozen[name], wording) for name in ("held", "calibration"))
    old = phrase_list((original, dev))
    old_raw, old_decoded = choices(base.encode(encoder, old), old, banks, kernel, scale)
    cuts = {arm: fit_cut(articles, dev, prediction) for arm, prediction in old_decoded.items()}
    cuts.update(strict_author=base.calibrate_roles(articles, dev, old_raw, False),
                agreement=base.calibrate_roles(articles, dev, old_raw, True))
    reference_names = dict(strict_author="strict_author_conditioned", agreement="bank_agreement",
                           ridge="query_ridge_calibrated", prototype="query_mixture_calibrated")
    for arm, key in reference_names.items():
        if not np.isclose(cuts[arm], archive["role_cutoffs"][key], atol=1e-6):
            raise ValueError(f"Changed original calibration: {arm}")
        cuts[arm] = archive["role_cutoffs"][key]
    bank, plans = stress_sets(args.wording, training, wording, original)
    new = sorted(phrase for rows in bank.values() for phrases in rows.values() for phrase in phrases)
    new_raw, new_decoded = choices(base.encode(encoder, new), new, banks, kernel, scale)
    reports = {}
    for mode in ("excerpt", "keywords"):
        proposals = list(prior.source_proposals(catalog, encoder, original, mode, args.reverse))
        control = quality(articles, original, proposals, old_raw, old_decoded, cuts)
        for track, rows in control.items():
            for arm, key in reference_names.items():
                expected = archive["elife"][mode]["counts"][track][key]
                if any(rows[arm][field] != expected[field] for field in ("strict", "wrong", "adverse_groups")):
                    raise ValueError(f"Original counts changed for {mode} {track} {arm}")
        reports[mode] = dict(niso=control, stress={
            view: quality(articles, plan, proposals, new_raw, new_decoded, cuts) for view, plan in plans.items()
        })
    base.write_json(args.output, dict(
        catalog_articles=len(articles), embedding=info["revision"], frozen_sha256=base.digest(args.frozen),
        reference_sha256=base.digest(args.reference), wording_sha256=base.digest(args.wording),
        script_sha256=base.digest(Path(__file__)), training_sha256=base.digest(args.training),
        kernel_scale=scale, kernel_oof=selection, cutoffs=cuts, original_recognition=prior.recognition(original, old_decoded),
        stress_recognition=recognition(bank, new_decoded), results=reports, gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
