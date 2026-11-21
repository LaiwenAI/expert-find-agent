"""Compare transferred and target-domain acceptance on a prospective PLOS split."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import torch

from experiments.credit import freeze_plos_open_transfer as frozen
from experiments.credit import plos_open_directed_transfer as transfer

route, base, role = transfer.route, transfer.base, transfer.role
POLICIES = ("elife_transfer", "plos_strict", "both_strict")


def restore(archive, holdout, path, older_projection, confirmation, extended):
    if base.digest(older_projection) != archive["projection_sha256"]:
        raise ValueError("Changed base PLOS distractor catalog")
    old = role.read(older_projection)
    excluded = {a["person"] for source in (confirmation, extended) for doc in role.read(source)["sources"]
                for a in doc["authors"] if a["person"]}
    projected, _ = frozen.load((path,), excluded, holdout["source_sha256"])
    if set(projected) != set(holdout["source_sha256"]) or set(old) & set(projected):
        raise ValueError("Changed or overlapping prospective PLOS documents")
    fresh = {}
    for name in ("calibration", "test"):
        for group in holdout[name]["pairs"]:
            people = set().union(*(projected[doc]["recorded"] for doc in group["papers"]))
            if any(people.intersection(previous) for previous in fresh.values()):
                raise ValueError("Prospective groups share a recorded author")
            fresh[tuple(group["papers"])] = people
    if not set(old).isdisjoint(paper for group in fresh for paper in group):
        raise ValueError("New groups used a base-catalog document")
    return old | {doc: {field: record[field] for field in ("indexed", "clue", "roles")}
                  for doc, record in projected.items()}


def target_cutoffs(rows, model, catalog):
    largest = {arm: defaultdict(lambda: -1.) for arm in route.METHODS}
    recovered = Counter()
    for papers, absent, cases in rows:
        for gold, labels, _, options in cases:
            for arm, (pairs, _, _) in options.items():
                for pair, score in route.scored(pairs, model):
                    if route.result_is_correct(catalog, papers, pair, gold, labels):
                        recovered[arm] += 1
                    else:
                        largest[arm][papers] = max(score, largest[arm][papers])
    cuts = {arm: max(values.values()) for arm, values in largest.items() if values}
    wrong_groups = {arm: len(values) for arm, values in largest.items()}
    return cuts, wrong_groups, recovered


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "elife_groups", "elife_held", "role_archive", "elife_policy",
                 "base_plos_manifest", "base_plos_projection", "new_plos_manifest", "plos_xml",
                 "glossary", "training", "embedding", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm GPU and a fresh output")
    prior = role.read(args.elife_groups)["panels"]
    role_archive = role.read(args.role_archive)
    model_archive = role.read(args.elife_policy)
    base_plos = role.read(args.base_plos_manifest)
    panel = role.read(args.new_plos_manifest)
    gpu = base.verify_checkpoint(args.embedding)
    if (gpu["revision"] != role_archive["embedding"] or base.digest(args.elife_held) != model_archive["panel_sha256"]
            or base.digest(args.elife_groups) != role_archive["frozen_sha256"]
            or tuple(panel["elife_sha256"]) != base.INPUTS
            or base.digest(args.role_archive) != model_archive["archive_sha256"]
            or base.digest(Path(route.__file__)) != model_archive["script_sha256"]):
        raise ValueError("Changed eLife calibration setup")
    elife = base.load_catalog((args.confirmation, args.extended))
    plos = restore(base_plos, panel, args.plos_xml, args.base_plos_projection, args.confirmation, args.extended)
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, training, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, training)
    weights = role_archive["training"]["weights"]
    role_cut = role_archive["role_cutoffs"]["query_ridge_calibrated"]
    elife_plans = tuple(base.phrase_cases(group, wording)
                        for group in (prior["calibration"], prior["held"], role.read(args.elife_held)))
    plos_plans = {name: transfer.plos_requests(plos, panel[name], wording) for name in ("calibration", "test")}
    _, elife_decision = role.embed(encoder, banks, weights, elife_plans)
    _, plos_decision = role.embed(encoder, banks, weights, tuple(plos_plans.values()))
    e_index = route.PostingIndex(route.Catalog(elife, encoder))
    p_index = route.PostingIndex(route.Catalog(plos, encoder))
    e_footprint = route.nested.previous.Fingerprints(e_index.catalog)
    p_footprint = route.nested.previous.Fingerprints(p_index.catalog)
    order = sorted(prior["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    e_dev, e_old, _ = elife_plans
    reports = {}
    for entrance in ("excerpt", "keywords"):
        def elife_rows(groups, *, training=False):
            return route.data(e_index, encoder, groups, e_dev | e_old, entrance,
                              elife_decision["ridge"], role_cut, e_footprint, args.reverse, training=training)

        def plos_rows(name):
            return route.data(p_index, encoder, panel[name]["pairs"], plos_plans[name], entrance,
                              plos_decision["ridge"], role_cut, p_footprint, args.reverse)

        scorer, training_counts = route.fit(elife_rows(order[:18], training=True))
        old_thresholds, adverse = route.calibrate(elife_rows(order[18:] + prior["held"]["pairs"]), scorer, elife)
        reference = model_archive["outcomes"][entrance]
        transfer.check_calibration(training_counts, old_thresholds, adverse, reference)
        base_cuts = reference["cutoffs"]
        local_cuts, target_adverse, attainable = target_cutoffs(plos_rows("calibration"), scorer, plos)
        target = base_cuts | local_cuts
        cuts = dict(elife_transfer=base_cuts, plos_strict=target,
                    both_strict={arm: max(base_cuts[arm], target[arm]) for arm in route.METHODS})
        held_rows = list(plos_rows("test"))
        reports[entrance] = dict(
            target_calibration_adverse_groups=target_adverse,
            target_calibration_correct_options=dict(attainable),
            target_threshold_available=sorted(local_cuts),
            thresholds=cuts,
            held={policy: route.evaluate(held_rows, scorer, cuts[policy], plos) for policy in POLICIES},
        )
    base.write_json(args.output, dict(
        plos_articles=len(plos), group_counts={name: len(plan) for name, plan in plos_plans.items()},
        role_phrase_recognition={name: role.recognition(plan, plos_decision) for name, plan in plos_plans.items()},
        reports=reports, manifest_sha256=base.digest(args.new_plos_manifest),
        script_sha256=base.digest(Path(__file__)), elife_policy_sha256=base.digest(args.elife_policy),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
