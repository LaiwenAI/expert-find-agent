"""Separate fixed-context attribution, competing mentions, and score calibration."""

import argparse
from collections import Counter
from itertools import groupby
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins
from experiments.docred import binding_marker_training as binding
from experiments.docred import person_bound_evidence as names
from experiments.docred import reference_preserving_window as windows
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("shared", "prefix_control", "scoped")


def scoped_rows(case, picked, slot):
    work, person = case["names"][slot]
    other_work = case["names"][slot ^ 2][0]
    other_person = case["names"][slot ^ 1][1]
    patterns = [binding.name_pattern([name]) for name in (work, person, other_work, other_person)]
    flags = [tuple(bool(pattern.search(case["sentences"][i])) for pattern in patterns) for i in picked]
    if not any(work_here and person_here for work_here, person_here, _, _ in flags):
        return picked
    kept = [i for i, (a, b, c, d) in zip(picked, flags, strict=True)
            if not ((c and not a) or (d and not b))]
    return kept or picked


def example(case, picked, reader):
    context = lambda rows: "\n".join(case["sentences"][i] for i in rows)
    prompts = [f"Did {person} have the role of {names.ROLES[case['role']]} on {work}?"
               for work, person in case["names"]]
    pairs = [(prompt, context(rows)) for prompt, rows in zip(prompts, picked, strict=True)]
    values, clipped = margins(reader, pairs, binding.TASK)
    return list(map(float, values)), clipped, sum(len(text.split()) for _, text in pairs)


def measure(cases, reader, offset):
    records = []
    for i, case in enumerate([cases[0], *cases]):
        shift = (i + offset) % len(ARMS)
        arms = {}
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            shared = windows.select(case, "global", None, None, None)
            if len(shared) != 4:
                raise ValueError("The shared input budget must be four")
            if arm == "shared":
                inputs = [shared] * 4
            else:
                scoped = [scoped_rows(case, shared, j) for j in range(4)]
                inputs = scoped if arm == "scoped" else [shared[:len(rows)] for rows in scoped]
            values, clipped, words = example(case, inputs, reader)
            json.dumps(dict(inputs=inputs, margins=values))
            arms[arm] = dict(scores=values, clipped=clipped, words=words, inputs=inputs,
                             ms=(perf_counter() - start) * 1000)
        if i:
            records.append((case, arms))
    return records


def development_cutoffs(rows):
    thresholds = {}
    for arm in ARMS:
        rivals = sorted((score for case, arms in rows
                         for score, known in zip(arms[arm]["scores"], case["labeled"], strict=True) if not known),
                        reverse=True)
        thresholds[arm] = rivals[len(rivals) // 20]
    return thresholds


def summarize(rows, cutoffs):
    groups = [case["doc"] for case, _ in rows]
    cap = sum(score > cutoffs["shared"] and not known for case, reads in rows
              for score, known in zip(reads["shared"]["scores"], case["labeled"], strict=True))
    output = {}
    reference = None
    for arm in ARMS:
        counts = Counter()
        successes = []
        cutoff = cutoffs[arm]
        for case, read in rows:
            choices = read[arm]
            accepted = [score > cutoff for score in choices["scores"]]
            fixed = [score > cutoffs["shared"] for score in choices["scores"]]
            supported = 0
            for j, (yes, fair, known) in enumerate(zip(accepted, fixed, case["labeled"], strict=True)):
                reference_present = bool(case["refs"][j]) and case["refs"][j] <= set(choices["inputs"][j])
                counts.update(accepted_known=int(yes and known), accepted_unjudged=int(yes and not known),
                              accepted_at_shared_cutoff_known=int(fair and known),
                              accepted_at_shared_cutoff_unjudged=int(fair and not known),
                              complete_refs=int(known and reference_present),
                              accepted_complete=int(yes and known and reference_present))
                supported += int(yes and known)
            counts["sentences"] += sum(map(len, choices["inputs"]))
            counts["words"] += choices["words"]
            counts["clipped"] += choices["clipped"]
            counts["pruned_claims"] += sum(len(ref) < 4 for ref in choices["inputs"])
            successes.append(supported)
        if reference is None:
            reference = successes
        output[arm] = dict(
            counts=dict(counts),
            p95_ms=float(np.quantile([arms[arm]["ms"] for _, arms in rows], .95)),
            known_delta_per_case=clustered_paired_interval(np.array(successes) - reference, groups),
            at_observed_shared_cap=matched_unjudged(rows, arm, cap),
        )
    return output


def matched_unjudged(rows, arm, cap):
    """Retrospective matched-count audit. Scores tied at a threshold enter together."""
    items = sorted(((score, known) for case, reads in rows
                    for score, known in zip(reads[arm]["scores"], case["labeled"], strict=True)),
                   reverse=True)
    known = unjudged = 0
    paired = dict(cap=cap, known=0, unjudged=0)
    for _, bucket in groupby(items, key=lambda item: item[0]):
        for _, supported in bucket:
            known += supported
            unjudged += not supported
        if unjudged > cap:
            break
        paired = dict(cap=cap, known=known, unjudged=unjudged)
    return paired


def oracle(rows, reader, cutoff, offset):
    counters = Counter()
    audit = []
    for case, arms in rows:
        shared = arms["shared"]
        for j, ref in enumerate(case["refs"]):
            if not case["labeled"][j] or not ref or not ref <= set(shared["inputs"][j]):
                continue
            extra = sorted(set(shared["inputs"][j]) - ref)
            views = {"reference": sorted(ref), "shared": shared["inputs"][j]}
            if extra:
                nearby = sorted((i for i in range(len(case["sentences"])) if i not in ref),
                                key=lambda i: (min(abs(i - s) for s in ref), i))[:len(extra)]
                views["nearby"] = sorted(ref | set(nearby))
                counters["nearby_changes_shared"] += views["nearby"] != views["shared"]
            observations = {}
            for name in list(views)[::1 if offset % 2 == 0 else -1]:
                scores, clipped, _ = example(case, [views[name]] * 4, reader)
                if clipped:
                    raise ValueError("Reference intervention was clipped")
                positive, rival = scores[j], scores[j ^ 1]
                observations[name] = dict(score=positive, rival=rival, rows=views[name])
                counters[name + "_compared"] += 1
                counters[name + "_zero"] += positive > 0
                counters[name + "_frozen"] += positive > cutoff
                counters[name + "_outranks_other_person"] += positive > rival
                if not case["labeled"][j ^ 1]:
                    counters[name + "_unjudged_rival_compared"] += 1
                    counters[name + "_outranks_unjudged_rival"] += positive > rival
            original = observations["shared"]["score"]
            if original != shared["scores"][j]:
                raise AssertionError("Reference audit did not reproduce the original shared score")
            passed = observations["reference"]["score"] > cutoff
            counters["reference_rescues"] += passed and original <= cutoff
            counters["reference_losses"] += not passed and original > cutoff
            audit.append(dict(case=case["id"], claim=j, competing_person_is_known=case["labeled"][j ^ 1],
                              views=observations))
    return dict(counters), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "expanded", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    train, dev, test = binding.panel(args.original, args.expanded)
    panels = {
        group: windows.add_annotations(rows, args.original / f"{split}_revised.json")
        for group, rows, split in (("calibration", dev, "dev"), ("held", test, "test"), ("extension", train, "train"))
    }
    reader = Reranker(args.reranker)
    records = {group: measure(rows, reader, args.offset) for group, rows in panels.items()}
    cutoffs = development_cutoffs(records["calibration"])
    oracle_outcomes = {}
    for key in ("held", "extension"):
        counts, trace = oracle(records[key], reader, cutoffs["shared"], args.offset)
        oracle_outcomes[key] = dict(counts=counts, evidence=trace)
    write_json(args.output, dict(
        panels={group: dict(cases=len(rows), titles=len({r["doc"] for r in rows})) for group, rows in panels.items()},
        cutoffs=cutoffs, results={group: summarize(rows, cutoffs) for group, rows in records.items()},
        oracle=oracle_outcomes, script_sha256=digest(Path(__file__)), offset=args.offset,
        gpu=torch.cuda.get_device_name(), reranker_revision=reader.provenance["revision"],
    ))


if __name__ == "__main__":
    main()
