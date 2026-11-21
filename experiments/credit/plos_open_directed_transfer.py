"""Apply the locked eLife directed recovery policy to a PLOS ONE catalog."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import torch

from experiments.credit import directed_source_rescue as route

base, role = route.base, route.role


def plos_requests(articles, panel, wording):
    available = defaultdict(list)
    for i, _, text in wording["niso"]:
        available[base.ROLES[i]].append(text)
    cases = {}
    for group in panel["pairs"]:
        papers = tuple(group["papers"])
        choices = []
        for i, pair in enumerate(group["contrasts"]):
            key = base.stable(papers, i)
            for label, roles in zip(("positive", "contrast"), pair, strict=True):
                if bool(base.known_owners(articles, papers, roles)) != (label == "positive"):
                    raise ValueError("Changed frozen role contrast")
                words = [min(available[name], key=lambda text: base.stable(key, name, text)) for name in roles]
                choices.append((label, roles, words))
        cases[papers] = choices
    return cases


def check_calibration(model_counts, cuts, adverse, reference):
    if dict(true=model_counts[True], wrong=model_counts[False]) != reference["training_counts"]:
        raise ValueError("Calibration training rows changed")
    if adverse != reference["adverse_calibration_groups"]:
        raise ValueError("Adverse calibration groups changed")
    for name in route.METHODS:
        if not np.isclose(cuts[name], reference["cutoffs"][name], atol=1e-6, rtol=0):
            raise ValueError(f"Frozen operating point did not reproduce: {name}")


def missed_recoveries(rows, model, cuts, articles):
    totals = {arm: Counter() for arm in route.METHODS}
    for papers, withheld, cases in rows:
        if withheld is not None:
            continue
        for gold, labels, _, options in cases:
            for arm, (pairs, _, _) in options.items():
                if papers not in pairs or not route.result_is_correct(articles, papers, papers, gold, labels):
                    continue
                total = totals[arm]
                total["reachable"] += 1
                scores = dict(route.scored(pairs, model))
                exact = scores[papers]
                winner = max(scores, key=scores.__getitem__)
                total["correct_below_cutoff"] += exact <= cuts[arm]
                total["wrong_ranks_higher"] += winner != papers
                total["wrong_accepted"] += winner != papers and scores[winner] > cuts[arm]
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("confirmation", "extended", "prior", "prior_held", "role_archive", "policy_archive", "plos_manifest",
                  "plos_projection", "glossary", "training", "embedding", "output"):
        p.add_argument(field, type=Path)
    p.add_argument("--source-supplied-panel", type=Path)
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm GPU allocation")
    checkpoint = base.verify_checkpoint(args.embedding)
    role_archive = role.read(args.role_archive)
    policy_archive = role.read(args.policy_archive)
    plos_manifest = role.read(args.plos_manifest)
    if (role_archive["embedding"] != checkpoint["revision"]
            or base.digest(args.prior_held) != policy_archive["panel_sha256"]
            or base.digest(args.role_archive) != policy_archive["archive_sha256"]
            or base.digest(args.prior) != role_archive["frozen_sha256"]
            or base.digest(args.plos_projection) != plos_manifest["projection_sha256"]
            or base.digest(Path(route.__file__)) != policy_archive["script_sha256"]):
        raise ValueError("Changed model, calibration, or held inputs")
    elife = base.load_catalog((args.confirmation, args.extended))
    plos = role.read(args.plos_projection)
    old = role.read(args.prior)["panels"]
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, training, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, training)
    weight = role_archive["training"]["weights"]
    threshold = role_archive["role_cutoffs"]["query_ridge_calibrated"]
    elife_plans = tuple(base.phrase_cases(value, wording)
                        for value in (old["calibration"], old["held"], role.read(args.prior_held)))
    _, elife_decode = role.embed(encoder, banks, weight, elife_plans)
    panels = {"new_groups": plos_manifest}
    unavailable_groups = 0
    if args.source_supplied_panel:
        if base.digest(args.source_supplied_panel) != plos_manifest["prior_plos_sha256"][0]:
            raise ValueError("Changed previously source-supplied PLOS panel")
        previous = role.read(args.source_supplied_panel)["panels"]["plos"]["pairs"]
        supported = [group for group in previous if set(group["papers"]) <= plos.keys()]
        if not supported:
            raise ValueError("None of the previously source-supplied groups is in the pinned catalog")
        panels["previously_source_supplied"] = dict(pairs=supported)
        unavailable_groups = len(previous) - len(supported)
    plans = {name: plos_requests(plos, panel, wording) for name, panel in panels.items()}
    plos_decode = {name: role.embed(encoder, banks, weight, (plan,))[1] for name, plan in plans.items()}
    elife_index = route.PostingIndex(route.Catalog(elife, encoder))
    plos_index = route.PostingIndex(route.Catalog(plos, encoder))
    elife_footprints = route.nested.previous.Fingerprints(elife_index.catalog)
    plos_footprints = route.nested.previous.Fingerprints(plos_index.catalog)
    order = sorted(old["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    dev, past, _ = elife_plans
    tracked = {}
    for mode in ("excerpt", "keywords"):
        def fit_rows(groups, *, training_set=False):
            return route.data(elife_index, encoder, groups, dev | past, mode,
                              elife_decode["ridge"], threshold, elife_footprints,
                              args.reverse, training=training_set)

        model, trained = route.fit(fit_rows(order[:18], training_set=True))
        cuts, adverse = route.calibrate(fit_rows(order[18:] + old["held"]["pairs"]), model, elife)
        archive = policy_archive["outcomes"][mode]
        check_calibration(trained, cuts, adverse, archive)
        tracked[mode] = {}
        for name, panel in panels.items():
            held = list(route.data(plos_index, encoder, panel["pairs"], plans[name], mode,
                                   plos_decode[name]["ridge"], threshold, plos_footprints, args.reverse))
            tracked[mode][name] = dict(
                outcomes=route.evaluate(held, model, archive["cutoffs"], plos),
                recovery_audit=missed_recoveries(held, model, archive["cutoffs"], plos),
            )
    base.write_json(args.output, dict(
        outcomes=tracked, catalog_articles=len(plos), held_groups={name: len(plan) for name, plan in plans.items()},
        depth=route.DEEP, prior_groups_unavailable=unavailable_groups,
        script_sha256=base.digest(Path(__file__)), manifest_sha256=base.digest(args.plos_manifest),
        frozen_policy_sha256=base.digest(args.policy_archive), embedding_revision=checkpoint["revision"],
        role_phrase_recognition={name: role.recognition(plan, plos_decode[name]) for name, plan in plans.items()},
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
