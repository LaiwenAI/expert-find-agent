"""Test source-first constraint verification on prose without provenance headers."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.numeric_evidence import constraint, evaluate_measurement
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge, LABELS
from experiments.transfer import owner_scope_counterfactual as parent


SELECT = (
    "Identify the single passage that reports an empirical measurement of the given metric in the exact "
    "setting, for the specified project, and credits the named person as test lead for that measurement. "
    "A report editor or observer is not a test lead. Ignore the comparator. Reply 0, 1, 2 or 3 for the "
    "zero-based index of the passage, or 4 if none qualifies."
)
CHECK = (
    "Does this passage alone report the requested empirical measurement and credit the specified person "
    "as the test lead for that measurement on the specified project and in the exact setting? "
    "An editor, observer, proposed result or measurement from a different setting does not qualify. "
    "Reply 1 for yes or 0 for no."
)
METHODS = ("direct", "direct_rotated", "select_then_compare", "select_verify_compare", "gold_card")


def narrative(text, style):
    owner, project, role, statement = (line.split(": ", 1)[1] for line in text.splitlines())
    action = "led the testing" if role == "test lead" else "edited the resulting report"
    if style == "plain":
        return f"{owner} {action} on the {project} project. {statement}"
    other = parent.OTHER if owner == parent.AUTHOR else parent.AUTHOR
    other_project = parent.OTHER_PROJECT if project == parent.PROJECT else parent.PROJECT
    return (f"{other} observed the work on the {project} project. {owner} also participated in the "
            f"{other_project} project. For the {project} project described in this passage, "
            f"{owner} {action}. {statement}")


def build():
    originals = list(parent.generate_cases())
    for case in originals:
        expected = None
        if case["edit"] == "intact":
            isolated = [i for i, card in enumerate(case["cards"])
                        if parent.structured(dict(case, cards=[card]), "joint")[1]]
            expected, = isolated
        for style in ("plain", "distractors"):
            yield dict(
                case_id=case["case_id"] + "-" + style, pair=case["pair"], edit=case["edit"], style=style,
                request=case["request"], scope=case["scope"], owner=case["owner"], project=case["project"],
                cards=[narrative(card, style) for card in case["cards"]], expected_card=expected,
                verdict=case["verdict"],
            )


def selection_result(case, source):
    if source is None:
        return "unknown"
    return evaluate_measurement(constraint(case["request"]), case["cards"][source])[0]


def analyze(rows):
    report = {}
    for style in ("plain", "distractors"):
        subset = [row for row in rows if row["style"] == style]
        selection = Counter()
        arms = {}
        for row in subset:
            gold, handled = row["expected_card"], row["selected"]
            selection.update(
                selected_correct_card=int(gold is not None and handled == gold),
                selected_wrong_card=int(handled is not None and handled != gold),
                correctly_selected_none=int(gold is None and handled is None),
                missed_valid_card=int(gold is not None and handled is None),
                wrong_card_rejected_by_check=int(handled is not None and handled != gold and not row["verified"]),
            )
        for arm in METHODS:
            counts = Counter()
            by_group = {}
            for row in subset:
                predicted = row["outcomes"][arm]
                counts.update(
                    correct=int(predicted == row["verdict"]),
                    intact_correct=int(row["edit"] == "intact" and predicted == row["verdict"]),
                    false_eligibility_without_evidence=int(row["edit"] != "intact" and predicted == "eligible"),
                    unsupported_decision=int(row["edit"] != "intact" and predicted != "unknown"),
                    correct_available_eligible=int(predicted == row["verdict"] == "eligible"),
                )
                if row["edit"] == "intact":
                    by_group.setdefault(row["pair"], []).append(predicted == row["verdict"])
            arms[arm] = dict(counts, both_intact_correct=sum(all(cases) for cases in by_group.values()),
                             cold_path_p95_ms=float(np.quantile([row["ms"][arm] for row in subset], .95)))
        report[style] = dict(queries=len(subset), card_selection=selection, decisions=arms)
    return report


def run(output):
    import torch

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    judge = Judge()
    cache = {}
    samples = []
    orders = (LABELS, (*LABELS[1:], LABELS[0]))
    for case in build():
        info = {key: case[key] for key in ("scope", "owner", "project")}
        token_key = case["pair"], tuple(case["cards"])
        if token_key not in cache:
            picked, selection_ms = judge(SELECT, dict(info, cards=case["cards"]), (0, 1, 2, 3, None))
            verified, verify_ms = (False, 0.) if picked is None else judge(
                CHECK, dict(info, passage=case["cards"][picked]), (False, True),
            )
            cache[token_key] = picked, selection_ms, verified, verify_ms
        picked, selection_ms, verified, verify_ms = cache[token_key]
        results, times = {}, {}
        for name, order in zip(METHODS, orders):
            instruction = parent.RULE + " Reply with one digit. " + " ".join(
                f"{i} means {label}." for i, label in enumerate(order)
            )
            results[name], times[name] = judge(instruction, dict(info, request=case["request"], cards=case["cards"]), order)
        start = perf_counter()
        results["select_then_compare"] = selection_result(case, picked)
        times["select_then_compare"] = selection_ms + (perf_counter() - start) * 1000
        results["select_verify_compare"] = results["select_then_compare"] if verified else "unknown"
        times["select_verify_compare"] = times["select_then_compare"] + verify_ms
        start = perf_counter()
        results["gold_card"] = selection_result(case, case["expected_card"])
        times["gold_card"] = (perf_counter() - start) * 1000
        samples.append(dict(case, selected=picked, verified=verified, outcomes=results, ms=times))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        parent_sha256=digest(Path(parent.__file__)), model_provenance_sha256=judge.sha,
        cases=len(samples), results=analyze(samples),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("build", "run"))
    cli.add_argument("output", type=Path)
    options = cli.parse_args()
    if options.command == "build":
        with options.output.open("x") as file:
            for case in build():
                file.write(json.dumps(case) + "\n")
    else:
        run(options.output)
