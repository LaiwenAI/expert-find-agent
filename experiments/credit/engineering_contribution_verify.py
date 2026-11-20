"""Test author-scoped verification across two contribution-statement formats."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("shared", "length_control", "scoped", "scoped_role_only")
VIEWS = ("exact", "described")
TASK = ("Check whether the shown author contribution declaration credits the named "
        "person for the requested activity. The same activity credited only to "
        "a different person does not support the target.")
ROLE_TASK = "Check whether the shown contribution declaration states the requested activity."
DESCRIPTIONS = (
    "formulated the main idea and research objectives",
    "maintained and organized the research data and accompanying metadata",
    "applied statistical or mathematical techniques to analyze the data",
    "secured financial support for the research",
    "performed experiments or collected evidence for the study",
    "designed the study methods or experimental procedures",
    "coordinated and managed execution of the project",
    "supplied the study materials, instruments, or computing facilities",
    "implemented or maintained software used in the research",
    "provided scientific oversight and leadership for the study",
    "checked the reproducibility of results or procedures",
    "prepared the figures and visual presentation of the results",
    "prepared the initial manuscript text",
    "critically reviewed and revised the manuscript",
)


def stable(*parts):
    return sha256("|".join(map(str, parts)).encode()).hexdigest()


def cases(documents):
    result = defaultdict(list)
    for pmc, row in sorted(documents.items()):
        for i, role in enumerate(ROLES):
            credited = [int(i) for i, fields in row["roles"].items() if role in fields]
            other = [int(i) for i, fields in row["roles"].items() if role not in fields]
            if not credited or not other:
                continue
            positive, rival = [min(group, key=lambda author: stable(pmc, role, author))
                               for group in (credited, other)]
            pair = sorted((positive, rival), key=lambda author: stable("order", pmc, role, author))
            result[row["journal"]].append(dict(
                article=pmc, role=role, candidates=pair, known=tuple(author == positive for author in pair),
                described=DESCRIPTIONS[i], aliases=row["aliases"], blocks=row["blocks"],
            ))
    return result


def observe(case, reader, arm, wording):
    brief = case["role"] if wording == "exact" else case["described"]
    blocks = case["blocks"]
    start = perf_counter()
    evidence, prompts = [], []
    for author in case["candidates"]:
        if arm == "shared":
            selected = blocks
        else:
            scoped = [block for block in blocks if author in block["owners"]]
            if not scoped:
                raise ValueError("No original clause for a fully declared candidate")
            selected = blocks[:len(scoped)] if arm == "length_control" else scoped
        alias = case["aliases"][str(author)]
        prompt = f"Requested contribution: {brief}."
        if arm != "scoped_role_only":
            prompt = f"Target author: {alias}. {prompt}"
        context = "\n".join(block["text"] for block in selected)
        prompts.append((prompt, context))
        evidence.append(len(selected))
    values, clipped = margins(reader, prompts, ROLE_TASK if arm == "scoped_role_only" else TASK)
    scores = [float(v) for v in values]
    json.dumps(scores)
    ms = (perf_counter() - start) * 1000
    return dict(scores=scores, ms=ms, clauses=sum(evidence), clipped=clipped)


def run(panel, reader, offset):
    output = {}
    for journal, rows in panel.items():
        output[journal] = {view: [] for view in VIEWS}
        for view in VIEWS:
            for i, row in enumerate([rows[0], *rows]):
                shift = (i + offset) % len(ARMS)
                values = {arm: observe(row, reader, arm, view) for arm in ARMS[shift:] + ARMS[:shift]}
                if i:
                    output[journal][view].append((row, values))
    return output


def cutoff(rows, arm):
    rivals = sorted((score for case, scores in rows
                     for score, known in zip(scores[arm]["scores"], case["known"], strict=True) if not known),
                    reverse=True)
    return rivals[len(rivals) // 20]


def summarize(rows, arm, threshold, shared_cutoff):
    counts = Counter()
    exact = []
    trace = []
    for case, obs in rows:
        values = obs[arm]["scores"]
        matches = [v > threshold for v in values]
        shared = [v > shared_cutoff for v in values]
        accepted = [i for i, value in enumerate(matches) if value]
        label = case["known"]
        counts.update(
            declared=sum(label[i] for i in accepted),
            wrong_declared=sum(not label[i] for i in accepted),
            declared_at_shared_threshold=sum(yes and known for yes, known in zip(shared, label, strict=True)),
            wrong_at_shared_threshold=sum(yes and not known for yes, known in zip(shared, label, strict=True)),
            ordered=int(values[label.index(True)] > values[label.index(False)]),
            clean=int(tuple(matches) == label),
            clauses=obs[arm]["clauses"], clipped=obs[arm]["clipped"],
        )
        exact.append(int(tuple(matches) == label))
        trace.append(dict(article=case["article"], role=case["role"], candidates=case["candidates"],
                          known=label, scores=values, accepted=matches))
    return dict(counts), exact, trace


def evaluate(measured):
    results = {}
    for target, development in (("frontiers", "sensors"), ("sensors", "frontiers")):
        results[target] = {}
        for view in VIEWS:
            fit, held = measured[development][view], measured[target][view]
            thresholds = {arm: cutoff(fit, arm) for arm in ARMS}
            baseline = summarize(held, "shared", thresholds["shared"], thresholds["shared"])[1]
            results[target][view] = {}
            for arm, threshold in thresholds.items():
                cal, _, _ = summarize(fit, arm, threshold, thresholds["shared"])
                counts, clean, trace = summarize(held, arm, threshold, thresholds["shared"])
                results[target][view][arm] = dict(
                    cutoff=threshold, calibration=cal, held=counts,
                    p95_ms=float(np.quantile([x[arm]["ms"] for _, x in held], .95)),
                    paired_clean_delta=clustered_paired_interval(
                        np.array(clean) - baseline, [case["article"] for case, _ in held],
                    ),
                    evidence=trace,
                )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    panel = cases(json.loads(args.panel.read_bytes()))
    if set(panel) != {"sensors", "frontiers"}:
        raise ValueError("Both statement layouts are required")
    reader = Reranker(args.reranker)
    results = evaluate(run(panel, reader, args.offset))
    write_json(args.output, dict(
        cohorts={name: dict(cases=len(rows), articles=len({r["article"] for r in rows}))
                 for name, rows in panel.items()}, results=results,
        panel_sha256=digest(args.panel), script_sha256=digest(Path(__file__)),
        offset=args.offset, gpu=torch.cuda.get_device_name(), reranker_revision=reader.provenance["revision"],
    ))
