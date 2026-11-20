"""Evaluate generalization to a full contribution taxonomy and scope distractors."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from string import ascii_uppercase
from time import perf_counter

import numpy as np

from experiments.credit.elife_role_normalization_search import NATIVE
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


DESCRIPTIONS = {
    "Conceptualization": "Formulates the overarching research aims, questions, or initial ideas.",
    "Data curation": "Manages, cleans, documents, and preserves research data for continued use.",
    "Formal analysis": "Applies formal analytical techniques to study data.",
    "Funding acquisition": "Obtains project research funding.",
    "Investigation": "Carries out the research and collects observations or experimental evidence.",
    "Methodology": "Develops or designs a research method, protocol, or model.",
    "Project administration": "Coordinates planning and execution of the research activities.",
    "Resources": "Supplies the study's materials, participants, equipment, or analysis resources.",
    "Software": "Develops, implements, or tests the research code.",
    "Supervision": "Oversees research work, provides leadership, or mentors researchers.",
    "Validation": "Checks the replication or reproducibility of research outcomes.",
    "Visualization": "Produces figures or other visual presentation of research results.",
    "Writing – original draft": "Writes the initial manuscript or substantive initial text.",
    "Writing – review & editing": "Critiques and revises the manuscript as part of the research team.",
    "outside": "The required claim exceeds what a contribution declaration alone can establish, or names no available role.",
}
DUTIES = {
    "Conceptualization": (
        "proposed the scientific question that motivated the entire project",
        "set out the research aims before a protocol was selected",
        "originated the central idea of the study",
    ),
    "Data curation": (
        "organized and documented the deposited datasets so others could reuse them",
        "maintained versioned records and repaired errors in the metadata",
        "cleaned the archived survey files and recorded how fields should be interpreted",
    ),
    "Formal analysis": (
        "used a statistical model to estimate associations in the recorded data",
        "computed confidence intervals for the study's reported effects",
        "applied mathematical techniques to synthesize the collected evidence",
    ),
    "Funding acquisition": (
        "obtained the external award that financed the investigation",
        "secured the research grant that paid for the study",
        "raised financial support specifically for this project",
    ),
    "Investigation": (
        "conducted the study procedures and collected the resulting observations",
        "personally took the readings during the planned trials",
        "carried out the measurements that formed the empirical results",
    ),
    "Methodology": (
        "developed the protocol the researchers followed during the investigation",
        "worked out the comparison procedure and its choice of controls",
        "designed the measurement method used by the experimental team",
    ),
    "Project administration": (
        "coordinated milestones, scheduling and handoffs across the research team",
        "managed when and where the research activities were carried out",
        "managed the project's timeline and operational coordination",
    ),
    "Resources": (
        "gave the team access to the specimens required by the study",
        "made the measurement equipment available to the investigators",
        "supplied the research materials used by the experimental team",
    ),
    "Software": (
        "wrote the program implementing the researchers' algorithm",
        "implemented the project's code and tested its functions",
        "coded the computational routines used for the study",
    ),
    "Supervision": (
        "oversaw the research staff and mentored the doctoral student",
        "provided scientific leadership and supervised the work of junior researchers",
        "acted as the senior mentor responsible for research oversight",
    ),
    "Validation": (
        "independently repeated the published workflow to check that it reproduced the result",
        "tested whether the reported findings could be replicated",
        "verified the reproducibility of the project's experimental outputs",
    ),
    "Visualization": (
        "created the plots used to present the findings in the publication",
        "designed the data figures that appeared in the final article",
        "made the graphical displays used to communicate the research results",
    ),
    "Writing – original draft": (
        "composed the first full version of the research manuscript",
        "produced the initial written account of the study for publication",
        "wrote the first draft of the paper before coauthor review",
    ),
    "Writing – review & editing": (
        "critically reviewed the coauthors' draft and rewrote disputed passages",
        "provided substantive comments and revisions to an existing manuscript",
        "edited the team's research paper in response to critical feedback",
    ),
    "outside": (
        "personally performed the experiment at 140 degrees Celsius under pilot-scale conditions",
        "designed a protocol that meets a named manufacturing standard",
        "wrote the manuscript alone without any coauthor input",
        "produced an effect estimate with a confidence interval narrower than 0.2",
        "developed code certified for deployment in a regulated clinical product",
        "replicated the results independently at a different institution with the same effect size",
        "secured at least ten million dollars of external funding",
        "can transfer this method to high-volume production without new data",
        "is currently available to work twenty hours per week",
        "holds the named patent on the analytical method",
        "has supervised at least five successful doctoral completions",
        "personally collected the observations before the 2019 cutoff",
        "supplied materials of a specified purity to every collaborating laboratory",
        "was solely responsible for cleaning all raw records before analysis",
    ),
}
CONFUSERS = {
    "Methodology": "designed the study procedure",
    "Investigation": "collected the measurements",
    "Writing – original draft": "written the initial manuscript",
    "Funding acquisition": "secured funding",
    "Software": "programmed the analysis",
}


def prompt(judge, requirement, menu, digits):
    task = ("Select the single contribution field that establishes the whole required claim. "
            "The text may mention duties that are optional or explicitly not required. "
            "A declaration does not establish additional quantities, special conditions, "
            "exclusive responsibility, or future availability. Choose outside whenever none of "
            "the offered fields establishes the whole required claim. Return only its letter.")
    view = dict(requirement=requirement, fields={
        digits[i]: dict(name=field, scope=DESCRIPTIONS[field]) for i, field in enumerate(menu)
    })
    return judge(task, view, tuple(menu))


def evaluate(judge, offset):
    maximum = len(DESCRIPTIONS)
    digits = ascii_uppercase[:maximum]
    token_ids = [judge.tokenizer.encode(char, add_special_tokens=False) for char in digits]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("Expected single choice tokens")
    judge.ids = [ids[0] for ids in token_ids]
    menu_types = {
        "five": tuple(role for role in ROLES if role in NATIVE.values()) + ("outside",),
        "fourteen": ROLES + ("outside",),
    }
    result = {}
    for schema, menu in menu_types.items():
        views = ("plain", "optional_distractor")
        arms = ("forward", "reverse", "agreement")
        counts = {view: {arm: Counter() for arm in arms} for view in views}
        timing = {view: {arm: [] for arm in arms} for view in views}
        for i, (field, phrases) in enumerate(DUTIES.items()):
            group = ("original_field" if field in NATIVE.values()
                     else "additional_field" if field != "outside" else "outside_contract")
            expected = field if field in menu else "outside"
            for j, phrase in enumerate(phrases):
                options = [text for role, text in CONFUSERS.items() if role != field]
                optional = options[(i + 2*j) % len(options)]
                pairs = (("plain", phrase), ("optional_distractor",
                         f"The candidate may have {optional}, but this is not required. "
                         f"What is required is that the candidate {phrase}."))
                for view, requirement in pairs:
                    orders = {"forward": menu, "reverse": menu[::-1]}
                    rotation = arms[:2] if (i + j + offset) % 2 == 0 else arms[1::-1]
                    estimates = {arm: prompt(judge, requirement, orders[arm], digits) for arm in rotation}
                    a, a_ms = estimates["forward"]
                    b, b_ms = estimates["reverse"]
                    estimates["agreement"] = (a if a == b else "outside", a_ms + b_ms)
                    for arm, (prediction, elapsed) in estimates.items():
                        counts[view][arm].update(
                            **{group + "_cases": 1, group + "_correct": int(prediction == expected),
                               group + "_mapped": int(prediction != "outside")}
                        )
                        timing[view][arm].append(elapsed)
        result[schema] = {
            view: {arm: dict(values, p95_ms=float(np.quantile(timing[view][arm], .95)))
                   for arm, values in scores.items()}
            for view, scores in counts.items()
        }
    return result


def main():
    import torch

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    judge = Judge()
    readings = evaluate(judge, args.order_offset)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), model_provenance_sha256=judge.sha,
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
        results=readings,
    ))


if __name__ == "__main__":
    main()
