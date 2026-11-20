"""Normalize paraphrased duties once before reading study-bound declarations."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit.study_bound_initials_label_order import TASK
from experiments.credit.study_clause_selection import ROLE_BY_TASK, parse_clauses
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


SCHEMA = {
    "design": "Conceived the study or designed how the experiments would be run.",
    "execution": "Performed the experiments and generated the experimental observations.",
    "analysis": "Analyzed or interpreted the collected data.",
    "writing": "Wrote or drafted substantive content for the paper.",
    "resources": "Supplied materials, reagents or analysis tools used in the study.",
    "outside": "No single listed field can establish this requested contribution.",
}
PHRASES = {
    "design": (
        "set the experimental comparison before any trials took place",
        "decided which controls and treatments the protocol would include",
        "drew up the experimental setup used to test the hypothesis",
        "worked out how to arrange the trials in the first place",
        "planned the experimental procedures before data collection",
        "determined how to structure the empirical test",
    ),
    "execution": (
        "personally ran the trials that generated the observations",
        "operated the experimental procedure during data collection",
        "carried out the actual bench tests",
        "performed the hands-on measurements from the trials",
        "executed the planned tests and recorded the raw results",
        "conducted the field trials specified in the protocol",
    ),
    "analysis": (
        "interpreted what the collected measurements showed",
        "calculated the reported statistics from the observations",
        "worked out patterns in the resulting dataset",
        "examined the observed values to draw the numerical conclusions",
        "processed the empirical results to estimate the reported effects",
        "assessed how the measured values differed across groups",
    ),
    "writing": (
        "authored the initial text submitted to the journal",
        "composed substantive sections of the submitted manuscript",
        "put the study's findings into the paper's narrative",
        "drafted the main article reporting the investigation",
        "wrote the text of the scholarly report",
        "prepared the prose of the first manuscript draft",
    ),
    "resources": (
        "made laboratory reagents available for the study",
        "supplied the team with the instruments used to analyze the measurements",
        "provided access to the experimental specimens",
        "furnished the materials needed to run the experiments",
        "gave the investigators use of an existing analysis tool",
        "supplied the research samples for the experiment",
    ),
    "outside": (
        "secured the grant that funded the work",
        "approved the funding budget as a department officer",
        "mentored the student outside the study without contributing to the experiments",
        "managed meeting calendars without doing or designing any trials",
        "arranged travel reimbursements for the project team",
        "deposited the accepted publication in an institutional archive",
        "approved the ethics submission in the role of an independent reviewer",
        "negotiated the institutional contract underwriting the project",
        "arranged conference registration for the presenting authors",
        "verified compliance paperwork as an independent administrator",
        "supervised payroll for the team without doing research",
        "provided language translation for an unrelated administrative form",
    ),
}
METHODS = ("direct_yes1", "direct_yes0", "one_forward", "one_reverse", "agreement")


def extended_ids(judge):
    ids = [judge.tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(SCHEMA))]
    if any(len(token) != 1 for token in ids):
        raise ValueError("Expected single output tokens")
    judge.ids = [token[0] for token in ids]


def normalize(judge):
    totals = {mode: Counter() for mode in ("one_forward", "one_reverse", "agreement")}
    predictions = {}
    task = ("Map the requested duty to one contribution field. Use outside if the duty does not "
            "follow from any listed field, or if more than one field would be required. Reply with "
            "the digit of the best field.")
    arms = {"one_forward": tuple(SCHEMA), "one_reverse": tuple(reversed(SCHEMA))}
    for expected, phrases in PHRASES.items():
        for phrase in phrases:
            row = {}
            for arm, labels in arms.items():
                choices = {str(i): SCHEMA[label] for i, label in enumerate(labels)}
                row[arm] = judge(task, dict(duty=phrase, fields=choices), labels)
            lhs, lhs_ms = row["one_forward"]
            rhs, rhs_ms = row["one_reverse"]
            row["agreement"] = (lhs if lhs == rhs else "outside", lhs_ms + rhs_ms)
            for arm, (guess, _) in row.items():
                totals[arm].update(
                    correct=int(guess == expected),
                    covered_correct=int(expected != "outside" and guess == expected),
                    covered_wrong_field=int(expected != "outside" and guess not in {expected, "outside"}),
                    covered_abstention=int(expected != "outside" and guess == "outside"),
                    outside_rejected=int(expected == "outside" and guess == expected),
                )
            predictions[phrase] = row
    return predictions, totals


def variant(case, role, ordinal):
    choices = PHRASES[role]
    offset = int(case["case_id"][:8], 16)
    return choices[(offset + ordinal) % len(choices)]


def run(fixture, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    data = json.loads(fixture.read_bytes())
    judge = Judge()
    extended_ids(judge)
    normal, phrase_counts = normalize(judge)
    totals = {arm: Counter() for arm in METHODS}
    timing = {arm: [] for arm in METHODS}
    query_time = {arm: [] for arm in METHODS[2:]}
    for index, case in enumerate(data["rows"]):
        expected_fields = [ROLE_BY_TASK[t] for t in case["tasks"]]
        valid = [variant(case, role, i) for i, role in enumerate(expected_fields)]
        invalid = [valid[0], variant(case, "outside", 2)]
        paired = {arm: [] for arm in METHODS}
        for mode, tasks in (("covered", valid), ("outside", invalid)):
            for arm in METHODS[2:]:
                query_time[arm].append(sum(normal[t][arm][1] for t in tasks))
            for requested in ((0, 1) if index % 2 else (1, 0)):
                report, initials = case["study_labels"][requested], case["initials"]
                note = case["notes"][requested]
                expected = mode == "covered" and requested == case["positive_index"]
                view = dict(report=report, initials=initials, required_tasks=tasks, reports={report: note})
                decisions = {
                    "direct_yes1": judge(TASK + " Reply 1 for yes or 0 for no.", view, (False, True)),
                    "direct_yes0": judge(TASK + " Reply 0 for yes or 1 for no.", view, (True, False)),
                }
                start = perf_counter()
                declared = {field for _, field, owners in parse_clauses(note) if initials in owners}
                if (set(expected_fields) <= declared) != (requested == case["positive_index"]):
                    raise ValueError("Fixture truth and literal bindings disagree")
                parse_ms = (perf_counter() - start) * 1000
                for arm in METHODS[2:]:
                    start = perf_counter()
                    fields = {normal[t][arm][0] for t in tasks}
                    passed = len(fields) == 2 and "outside" not in fields and fields <= declared
                    decisions[arm] = passed, parse_ms + (perf_counter() - start) * 1000
                for arm, (passed, elapsed) in decisions.items():
                    timing[arm].append(elapsed)
                    totals[arm].update(positive_accept=int(expected and passed),
                                       in_schema_wrong_accept=int(mode == "covered" and not expected and passed),
                                       outside_wrong_accept=int(mode == "outside" and passed))
                    if mode == "covered":
                        paired[arm].append(passed == expected)
        for arm, outcomes in paired.items():
            totals[arm].update(both_correct=int(all(outcomes)))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), fixture_sha256=digest(fixture),
        model_provenance_sha256=judge.sha, source_pairs=data["source_pairs"],
        covered_phrases=sum(len(phrases) for k, phrases in PHRASES.items() if k != "outside"),
        outside_phrases=len(PHRASES["outside"]), phrase_mapping=phrase_counts,
        report_outcomes=totals,
        per_report_p95_ms={arm: float(np.quantile(values, .95)) for arm, values in timing.items()},
        once_per_query_p95_ms={arm: float(np.quantile(values, .95)) for arm, values in query_time.items()},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.fixture, args.output)
