"""Resolve each duty to a clause and require exact author binding."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.study_bound_initials_label_order import TASK
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge

CLAUSE_TASK = (
    "Select one contribution clause from the specified report that explicitly credits the given "
    "initials with the requested task. Choose NONE if no such clause exists. Output only its digit."
)
ROLE_BY_TASK = {phrases[1]: role for role, phrases in legacy.PHRASES.items()}
ARMS = ("direct_yes1", "direct_yes0", "report_0", "report_end", "report_agree",
        "author_0", "author_end", "author_agree", "literal")


def parse_clauses(note):
    clauses = []
    for sentence in legacy.SENTENCE.split(note):
        title, separator, credit = sentence.partition(":")
        people = credit.strip().rstrip(".").split()
        if separator and people and all(legacy.INITIAL.fullmatch(person) for person in people):
            clauses.append((sentence, legacy.LABELS.get(title.strip().casefold()), set(people)))
    return clauses


def pick(judge, report, initials, task, clauses, scoped, abstain_first):
    eligible = [i for i, (_, _, owners) in enumerate(clauses) if not scoped or initials in owners]
    if not eligible:
        return None, 0., False
    order = [None] + eligible if abstain_first else eligible + [None]
    if len(order) > len(judge.ids):
        raise ValueError("Too many contribution clauses for one digit per choice")
    view = dict(report=report, initials=initials, required_task=task,
                choices={str(i): "NONE" if cid is None else clauses[cid][0]
                         for i, cid in enumerate(order)})
    answer, elapsed = judge(CLAUSE_TASK, view, tuple(order))
    if answer is not None and initials not in clauses[answer][2]:
        return None, elapsed, True
    return answer, elapsed, False


def complete_claim(selected):
    return None not in selected and len(set(selected)) == len(selected)


def evaluate_view(judge, case, requested, totals):
    started = perf_counter()
    report, initials = case["study_labels"][requested], case["initials"]
    note = case["notes"][requested]
    tasks = case["tasks"]
    parts = parse_clauses(note)
    parse_ms = (perf_counter() - started) * 1000
    roles = [ROLE_BY_TASK[t] for t in tasks]
    truth = [{i for i, (_, role, owners) in enumerate(parts)
              if role == needed and initials in owners} for needed in roles]
    direct = dict(report=report, initials=initials, required_tasks=tasks, reports={report: note})
    verdicts = {
        "direct_yes1": judge(TASK + " Reply 1 for yes or 0 for no.", direct, (False, True)),
        "direct_yes0": judge(TASK + " Reply 0 for yes or 1 for no.", direct, (True, False)),
        "literal": (all(map(bool, truth)), parse_ms),
    }
    for group, scoped in (("report", False), ("author", True)):
        selections = {}
        for encoding, first in (("0", True), ("end", False)):
            arm = f"{group}_{encoding}"
            selected, ms, rejected = zip(*(
                pick(judge, report, initials, task, parts, scoped, first) for task in tasks
            ), strict=True)
            selections[encoding] = (selected, sum(ms), rejected)
            verdicts[arm] = (complete_claim(selected), parse_ms + sum(ms))
            summarize_roles(totals[arm], selected, truth, rejected)
        lead, lead_ms, lead_rejected = selections["0"]
        tail, tail_ms, tail_rejected = selections["end"]
        common = tuple(a if a == b else None for a, b in zip(lead, tail, strict=True))
        arm = f"{group}_agree"
        verdicts[arm] = (complete_claim(common), parse_ms + lead_ms + tail_ms)
        summarize_roles(totals[arm], common, truth,
                        tuple(a or b for a, b in zip(lead_rejected, tail_rejected, strict=True)))
    return verdicts, all(map(bool, truth))


def summarize_roles(counts, selected, truth, rejected):
    for answer, acceptable, wrong_owner in zip(selected, truth, rejected, strict=True):
        counts.update(
            role_lookups=1, role_pointer_correct=int(answer in acceptable),
            wrong_role_pointer=int(answer is not None and answer not in acceptable),
            no_clause=int(answer is None), rejected_wrong_owner=int(wrong_owner),
        )


def run(fixture, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    data = json.loads(fixture.read_bytes())
    judge = Judge()
    judge.ids = []
    for digit in range(10):
        raw = judge.tokenizer.encode(str(digit), add_special_tokens=False)
        if len(raw) != 1:
            raise ValueError("Choice labels must be single tokens")
        judge.ids.append(raw[0])
    totals = {arm: Counter() for arm in ARMS}
    timing = {arm: [] for arm in ARMS}
    for i, case in enumerate(data["rows"]):
        outcomes = {arm: [] for arm in ARMS}
        for requested in ((0, 1) if i % 2 else (1, 0)):
            observed, parsed_truth = evaluate_view(judge, case, requested, totals)
            expected = requested == case["positive_index"]
            if expected != parsed_truth:
                raise ValueError("Legacy statement parsing and frozen fixture disagree")
            for arm, (passed, elapsed) in observed.items():
                timing[arm].append(elapsed)
                totals[arm].update(accepted_support=int(passed and expected),
                                   unsupported_accepted=int(passed and not expected))
                outcomes[arm].append(passed == expected)
        for arm, pair in outcomes.items():
            totals[arm].update(both_correct=int(all(pair)))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        model_provenance_sha256=judge.sha, source_pairs=data["source_pairs"],
        results={key: dict(totals[key], p95_ms=float(np.quantile(timing[key], .95))) for key in ARMS},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.fixture, args.output)
