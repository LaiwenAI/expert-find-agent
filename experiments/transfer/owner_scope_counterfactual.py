"""Audit jointly bound ownership, project scope, and quantitative evidence."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from expert_search.numeric_evidence import constraint, evaluate_measurement
from experiments.transfer.constraint_counterfactual import Judge, LABELS, generate
from experiments.followir.core17_reranking import digest, write_json


AUTHOR = "Avery"
OTHER = "Morgan"
PROJECT = "Harbor"
OTHER_PROJECT = "Summit"
RULE = (
    "Decide eligibility under the request. Require a reported empirical measurement of the requested "
    "metric in the specified setting, and require the named person to be credited as lead test operator "
    "for that measurement in the specified project. An author on another project, another person's "
    "measurement, a different role, or a planned target cannot establish eligibility. "
    "Eligible means the reported measurement meets the bound. Reject means it violates the bound. "
    "Unknown means qualifying evidence is absent."
)
LABEL_ORDERS = {"direct_4b": LABELS, "direct_4b_rotated": (*LABELS[1:], LABELS[0])}
STRUCTURED = ("ignore_owner", "ignore_project", "joint")
ARMS = (*LABEL_ORDERS, *STRUCTURED)


def card(owner, project, duty, statement):
    return (
        f"Contributor: {owner}\nProject: {project}\nRole: {duty}\n"
        f"Statement: {statement}"
    )


def generate_cases():
    rng = random.Random(20261120)
    for item in generate():
        if item["view"] != "intact":
            continue
        measurement = item["cards"][item["gold_source"]]["text"]
        target = next(row["text"] for row in item["cards"] if row["role"] == "target")
        off_scope = next(row["text"] for row in item["cards"] if row["role"] == "off_scope")
        query = (
            f"Find evidence that {AUTHOR} personally led testing for Project {PROJECT}. "
            + item["query"]
        )
        pending = f"The pilot plans to measure {item['scope']['metric']} in {item['scope']['setting']}. Results are pending."
        frames = {
            "intact": (AUTHOR, PROJECT, "test lead", measurement),
            "wrong_owner": (OTHER, PROJECT, "test lead", measurement),
            "wrong_project": (AUTHOR, OTHER_PROJECT, "test lead", measurement),
            "wrong_role": (AUTHOR, PROJECT, "report editor", measurement),
            "planned_only": (AUTHOR, PROJECT, "test lead", pending),
        }
        order = rng.sample(range(4), 4)
        for edit, fields in frames.items():
            records = [
                card(*fields),
                card(AUTHOR, PROJECT, "test lead", off_scope),
                card(AUTHOR, PROJECT, "test lead", target),
                card(OTHER, OTHER_PROJECT, "test lead", measurement),
            ]
            yield dict(
                case_id=f"{item['id']}-{edit}", pair=item["pair"], threshold=item["id"],
                edit=edit, request=query, scope=item["scope"], owner=AUTHOR, project=PROJECT,
                cards=[records[i] for i in order],
                verdict=item["gold"] if edit == "intact" else "unknown",
            )


def structured(case, mode):
    limit = constraint(case["request"])
    required_owner, required_project = case["owner"], case["project"]
    verdicts = set()
    has_citation = False
    for text in case["cards"]:
        who, which, role, statement = (line.split(": ", 1)[1] for line in text.splitlines())
        if (mode != "ignore_owner" and who != required_owner
                or mode != "ignore_project" and which != required_project
                or role != "test lead"):
            continue
        if (case["scope"]["setting"] not in statement or case["scope"]["metric"] not in statement
                or not any(cue in statement for cue in ("measured", "observed", "recorded"))
                or "proposal" in statement):
            continue
        verdict, quote = evaluate_measurement(limit, statement)
        if quote:
            verdicts.add(verdict)
            has_citation |= who == required_owner and which == required_project
    return next(iter(verdicts)) if len(verdicts) == 1 else "unknown", has_citation


def summarize(rows):
    report = {}
    for arm in ARMS:
        groups = defaultdict(Counter)
        thresholds = defaultdict(list)
        times = []
        for row in rows:
            verdict, citation, ms = row["arms"][arm]
            gold, edit = row["verdict"], row["edit"]
            groups[edit].update(
                cases=1, correct=int(verdict == gold),
                false_eligibility=int(verdict == "eligible" and gold != "eligible"),
                unsupported_decisions=int(verdict != "unknown" and gold == "unknown"),
                supported_eligible=int(verdict == gold == "eligible" and citation is True),
            )
            if edit == "intact":
                thresholds[row["pair"]].append(verdict == gold)
            times.append(ms)
        report[arm] = dict(
            by_edit=groups, intact_pairs=len(thresholds),
            both_intact_correct=sum(len(m) == 2 and all(m) for m in thresholds.values()),
            warm_p95_ms=float(np.quantile(times, .95)),
        )
    return report


def run(fixture, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    rows = json.loads(fixture.read_bytes())
    if rows != list(generate_cases()):
        raise ValueError("Fixture does not match the deterministic generator")
    judge = Judge()
    prompts = {
        name: RULE + " Reply with one digit. " + " ".join(f"{i} means {label}." for i, label in enumerate(order))
        for name, order in LABEL_ORDERS.items()
    }
    arms = []
    for case in rows:
        model_view = {key: case[key] for key in ("request", "scope", "owner", "project", "cards")}
        outcomes = {}
        for name, order in LABEL_ORDERS.items():
            verdict, ms = judge(prompts[name], model_view, order)
            outcomes[name] = verdict, None, ms
        for name in STRUCTURED:
            start = perf_counter()
            verdict, citation = structured(case, name)
            outcomes[name] = verdict, citation, (perf_counter() - start) * 1000
        arms.append(dict(case_id=case["case_id"], pair=case["pair"], edit=case["edit"],
                         verdict=case["verdict"], arms=outcomes))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)), model_provenance_sha256=judge.sha,
        cases=len(rows), results=summarize(arms),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    gen = commands.add_parser("build")
    gen.add_argument("fixture", type=Path)
    replay = commands.add_parser("run")
    replay.add_argument("fixture", type=Path)
    replay.add_argument("output", type=Path)
    options = cli.parse_args()
    if options.mode == "build":
        with options.fixture.open("x") as file:
            file.write("[\n")
            file.write(",\n".join(json.dumps(row, separators=(",", ":")) for row in generate_cases()))
            file.write("\n]\n")
    else:
        run(options.fixture, options.output)
