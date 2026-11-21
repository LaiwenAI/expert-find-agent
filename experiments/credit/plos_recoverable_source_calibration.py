"""Expose recoverable and absent sources before calibrating rescue acceptance."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import torch

from experiments.credit import plos_prospective_calibration as prior

route, base, role = prior.route, prior.base, prior.role
SHALLOW, DEEP = route.SHALLOW, route.DEEP


def demote(topic, side, paper):
    """Remove one designated paper only from the shallow prefix on its side."""
    choices = topic[side]["fused"]
    if paper not in choices[:SHALLOW]:
        return None
    revised = list(topic)
    demoted = [p for p in choices if p != paper]
    demoted.insert(SHALLOW, paper)
    revised[side] = topic[side] | {"fused": demoted}
    if set(demoted) != set(choices) or demoted[SHALLOW] != paper:
        raise ValueError("Intervention changed the deep pool")
    return revised


def exposed(index, encoder, groups, plan, mode, predictions, role_cut, footprints, reverse):
    for group in sorted(groups, key=lambda row: row["papers"], reverse=reverse):
        pair = tuple(group["papers"])
        clues = [index.catalog.articles[doc]["clue"] for doc in pair]
        if mode == "keywords":
            clues = [index.catalog.keywords(text) for text in clues]
        encoded = encoder.encode(base.search.topic_prompts(clues))
        natural = index.catalog.rank(encoded, clues, None, depth=DEEP)
        for side, paper in enumerate(pair):
            modified = demote(natural, side, paper)
            if modified is not None:
                rows = list(route.choices(index, modified, clues, None, plan[pair],
                                          predictions, role_cut, footprints, False))
                yield pair, None, rows


def lexical_guard(model):
    # Keep the eLife fit, with a fixed extra half unit of logit support
    # divided over the weakest source's standardized lexical features.
    center, scale, beta = model
    coefficients = beta.copy()
    for feature in (3, 4, 6):
        coefficients[feature + 1] += .5 / 3
    return center, scale, coefficients


def calibration(rows, model, articles):
    per_group = defaultdict(dict)
    possible, positives = Counter(), Counter()
    for papers, absent, cases in rows:
        for gold, labels, _, options in cases:
            for arm, (pairs, _, _) in options.items():
                for pair, score in route.scored(pairs, model):
                    if absent is None and route.result_is_correct(articles, papers, pair, gold, labels):
                        possible[arm] += 1
                        positives[papers, arm] = max(positives[papers, arm], score)
                    else:
                        failures = per_group[arm]
                        failures[papers] = max(score, failures.get(papers, -1.))
    cuts = {arm: max(values.values()) for arm, values in per_group.items() if values}
    adverse = {arm: len(per_group[arm]) for arm in route.METHODS}
    supported_groups = {arm: sum(name == arm for _, name in positives) for arm in route.METHODS}
    separated = {arm: sum(score > cuts.get(arm, -1.) for (_, name), score in positives.items() if name == arm)
                 for arm in route.METHODS}
    return cuts, dict(adverse_groups=adverse, correct_options=dict(possible), positive_groups=supported_groups,
                      positive_groups_above_wrong_max=separated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "elife_groups", "elife_held", "role_archive", "elife_policy",
                 "base_plos_manifest", "base_plos_projection", "prospective_plos", "prospective_xml",
                 "prospective_result", "glossary", "training", "embedding", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm GPU")
    original = role.read(args.elife_groups)["panels"]
    role_ref, policy_ref = role.read(args.role_archive), role.read(args.elife_policy)
    plos_ref = role.read(args.prospective_result)
    panel = role.read(args.prospective_plos)
    metadata = base.verify_checkpoint(args.embedding)
    if (plos_ref["manifest_sha256"] != base.digest(args.prospective_plos)
            or plos_ref["elife_policy_sha256"] != base.digest(args.elife_policy)
            or role_ref["embedding"] != metadata["revision"]
            or base.digest(args.elife_groups) != role_ref["frozen_sha256"]
            or base.digest(args.elife_held) != policy_ref["panel_sha256"]
            or base.digest(Path(route.__file__)) != policy_ref["script_sha256"]):
        raise ValueError("Changed external or eLife inputs")
    e_articles = base.load_catalog((args.confirmation, args.extended))
    p_articles = prior.restore(role.read(args.base_plos_manifest), panel, args.prospective_xml,
                              args.base_plos_projection, args.confirmation, args.extended)
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, phrases, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, phrases)
    weights, role_cut = role_ref["training"]["weights"], role_ref["role_cutoffs"]["query_ridge_calibrated"]
    e_plan = tuple(base.phrase_cases(group, wording)
                   for group in (original["calibration"], original["held"], role.read(args.elife_held)))
    p_plan = {part: prior.transfer.plos_requests(p_articles, panel[part], wording)
              for part in ("calibration", "test")}
    _, e_roles = role.embed(encoder, banks, weights, e_plan)
    _, p_roles = role.embed(encoder, banks, weights, tuple(p_plan.values()))
    e_index = route.PostingIndex(route.Catalog(e_articles, encoder))
    p_index = route.PostingIndex(route.Catalog(p_articles, encoder))
    e_footprint, p_footprint = (route.nested.previous.Fingerprints(index.catalog) for index in (e_index, p_index))
    order = sorted(original["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    outcomes = {}
    for entrance in ("excerpt", "keywords"):
        e_training = route.data(e_index, encoder, order[:18], e_plan[0], entrance, e_roles["ridge"],
                                role_cut, e_footprint, args.reverse, training=True)
        e_tuning = list(route.data(e_index, encoder, order[18:] + original["held"]["pairs"], e_plan[0] | e_plan[1],
                                   entrance, e_roles["ridge"], role_cut, e_footprint, args.reverse))
        model, samples = route.fit(e_training)
        original_cut, adverse = route.calibrate(e_tuning, model, e_articles)
        prior.transfer.check_calibration(samples, original_cut, adverse, policy_ref["outcomes"][entrance])

        def rows(name, generator=route.data):
            return list(generator(p_index, encoder, panel[name]["pairs"], p_plan[name], entrance, p_roles["ridge"],
                                  role_cut, p_footprint, args.reverse))

        natural, rescued = rows("calibration"), rows("calibration", exposed)
        held, challenge = rows("test"), rows("test", exposed)
        guard = lexical_guard(model)
        guard_cut, _ = route.calibrate(e_tuning, guard, e_articles)
        natural_cut, natural_info = calibration(natural, model, p_articles)
        exposure_cut, exposure_info = calibration([*natural, *rescued], model, p_articles)
        lexical_cut, lexical_info = calibration([*natural, *rescued], guard, p_articles)
        policies = {
            "elife": (model, original_cut),
            "natural_plos": (model, original_cut | natural_cut),
            "paired_plos": (model, original_cut | exposure_cut),
            "paired_lexical_guard": (guard, guard_cut | lexical_cut),
        }
        evaluated = {}
        for name, (scorer, threshold) in policies.items():
            evaluated[name] = dict(
                threshold=threshold,
                natural=route.evaluate(held, scorer, threshold, p_articles),
                shallow_source_demoted=route.evaluate(challenge, scorer, threshold, p_articles)["present"],
            )
        if (evaluated["elife"]["natural"] != plos_ref["reports"][entrance]["held"]["elife_transfer"]
                or evaluated["natural_plos"]["natural"] != plos_ref["reports"][entrance]["held"]["plos_strict"]):
            raise ValueError("Archived natural controls did not reproduce")
        outcomes[entrance] = dict(
            intervention_calibration_trials=len(rescued), intervention_test_trials=len(challenge),
            calibration=dict(natural=natural_info, paired=exposure_info, lexical_guard=lexical_info),
            tested=evaluated,
        )
    base.write_json(args.output, dict(
        outcomes=outcomes, panel_sha256=base.digest(args.prospective_plos),
        reference_sha256=base.digest(args.prospective_result), script_sha256=base.digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
