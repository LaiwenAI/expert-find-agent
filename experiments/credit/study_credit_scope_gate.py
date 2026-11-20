"""Separate normalization of contribution roles from proof obligations."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from string import ascii_uppercase
from time import perf_counter

import numpy as np

from experiments.credit import study_credit_taxonomy_transfer as credit
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


CHECKS = {
    "role_only": "The required claim is only that the person performed a contribution type.",
    "measurement": "Requires an extra measurement, minimum quantity, date, operating condition, or demonstrated outcome.",
    "scope": "Requires a specified project, identity, exclusive responsibility, source provenance, or independent confirmation.",
    "other": "Requires other qualifications, a future commitment, or cannot be decided without further evidence.",
}
SURFACE = re.compile(
    r"\d|\b(?:alone|sole(?:ly)?|only|without|at least|no more than|certif\w*|"
    r"independent\w*|same project|specified|named|accredit\w*|prior to)\b", re.I,
)

# The same base role is used with an extra mandatory or explicitly optional claim.
CASES = (
    ("Investigation", "ran the trials that produced the data",
     "documented that the testing occurred on a production line rather than in a simulation"),
    ("Investigation", "carried out the experimental procedures",
     "ran the equipment continuously for at least 24 hours"),
    ("Investigation", "collected the empirical measurements",
     "took every reading in the specified test chamber"),
    ("Methodology", "designed the research procedure",
     "used the design on the previous contract under the same project identifier"),
    ("Methodology", "specified the experimental protocol",
     "demonstrated that the procedure remained valid at industrial scale"),
    ("Methodology", "developed the study method",
     "obtained independent certification against the required protocol standard"),
    ("Formal analysis", "computed the estimates reported in the study",
     "obtained a mean error below 0.1 on the external validation sample"),
    ("Formal analysis", "applied statistical methods to the study data",
     "ran the calculations using observations from the named production batch"),
    ("Formal analysis", "analyzed the measured outcomes",
     "showed the numerical result held after changing the operating conditions"),
    ("Writing – original draft", "wrote the initial research manuscript",
     "wrote the entire draft without substantive input from coauthors"),
    ("Writing – review & editing", "revised the results text after team review",
     "made all revisions after the editor's final decision"),
    ("Resources", "supplied the study equipment",
     "provided instruments with calibration valid through the planned start date"),
    ("Resources", "supplied materials used in the study",
     "supplied material of at least 99 percent purity"),
    ("Funding acquisition", "secured financial support for the project",
     "raised the funds for the follow-up study rather than the original report"),
    ("Software", "implemented the software used in the study",
     "tested the deployed version under peak load"),
    ("Software", "wrote and tested the analysis code",
     "was the only developer responsible for every release"),
    ("Validation", "checked whether the results could be reproduced",
     "replicated the measurements at a separate site before the stated deadline"),
    ("Supervision", "supervised the graduate researchers",
     "can supervise a new project starting next month"),
)
MODES = ("standalone", "mandatory_extra", "optional_extra")
ARMS = ("role_menu", "surface_guard", "binary_forward", "binary_reverse", "binary_guard",
        "proof_type_forward", "proof_type_reverse", "proof_type_guard")


def available(judge):
    letters = ascii_uppercase[:len(credit.DESCRIPTIONS)]
    labels = [judge.tokenizer.encode(symbol, add_special_tokens=False) for symbol in letters]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Expected choice letters to be single tokens")
    judge.ids = [token[0] for token in labels]
    return letters


def scope_pick(judge, text, menu, letters, detailed):
    task = ("Assume a correct declaration identifies the person's contribution type in the paper. "
            "Decide whether that field alone can establish the entire mandatory request. "
            "Ignore optional extras, and require separate proof for mandatory conditions. "
            "If uncertain use the conservative choice. Reply with the letter only.")
    categories = CHECKS if detailed else {
        "role_only": "The declaration alone can establish the entire mandatory request.",
        "other": "An extra mandatory condition or uncertainty needs separate proof.",
    }
    view = dict(requirement=text, choices={letters[i]: categories[label] for i, label in enumerate(menu)})
    return judge(task, view, menu)


def decisions(handler, menu, offset):
    orders = (("forward", menu), ("reverse", menu[::-1]))
    if offset % 2:
        orders = tuple(reversed(orders))
    output = {name: handler(options) for name, options in orders}
    left, right = output["forward"], output["reverse"]
    outcome = left[0] if left[0] == right[0] else None
    output["guard"] = (outcome, left[1] + right[1])
    return output


def evaluate(judge, offset):
    letters = available(judge)
    first = credit.ROLES + ("outside",)
    counts = {mode: {arm: Counter() for arm in ARMS} for mode in MODES}
    timing = {mode: {arm: [] for arm in ARMS} for mode in MODES}
    for i, (role, base, extra) in enumerate(CASES):
        rows = (f"The candidate {base}.",
                f"The candidate {base}. The candidate also must meet this condition: they {extra}.",
                f"The candidate {base}. This additional condition is optional: they {extra}.")
        for mode, text in zip(MODES, rows, strict=True):
            role_result, role_ms = decisions(
                lambda menu: credit.prompt(judge, text, menu, letters), first, i + offset,
            )["guard"]
            gate_results = {}
            for family, detailed in (("binary", False), ("proof_type", True)):
                labels = tuple(CHECKS) if detailed else ("role_only", "other")
                result = decisions(
                    lambda menu: scope_pick(judge, text, menu, letters, detailed),
                    labels, i + offset,
                )
                for encoding, (decision, elapsed) in result.items():
                    gate_results[f"{family}_{encoding}"] = (decision == "role_only", elapsed)
            begin = perf_counter()
            passes_surface = SURFACE.search(text) is None
            surface_ms = (perf_counter() - begin) * 1000
            for arm, (gate_pass, gate_ms) in (
                ("role_menu", (True, 0.)), ("surface_guard", (passes_surface, surface_ms)),
                *gate_results.items(),
            ):
                issued = role_result not in (None, "outside") and gate_pass
                counts[mode][arm].update(
                    cases=1, issued=int(issued), correct_role_issued=int(issued and role_result == role),
                    wrong_role_issued=int(issued and role_result != role),
                    extra_proof_bypassed=int(issued and mode == "mandatory_extra"),
                    held_for_proof=int(not issued),
                )
                timing[mode][arm].append(role_ms + gate_ms)
    return {mode: {arm: dict(values, p95_ms=float(np.quantile(timing[mode][arm], .95)))
                   for arm, values in arms.items()} for mode, arms in counts.items()}


def main():
    import torch

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    judge = Judge()
    report = evaluate(judge, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), decision_model_sha256=judge.sha, results=report,
    ))


if __name__ == "__main__":
    main()
