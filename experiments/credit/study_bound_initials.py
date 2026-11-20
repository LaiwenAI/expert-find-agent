"""Distinguish same-initial contributors across authentic publisher statements."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import legacy_contribution_text as legacy
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


TASK = (
    "Judge whether the contributor with the specified initials is explicitly credited with BOTH required "
    "tasks in the specified report. Report labels are distinct. Identical initials in another report "
    "cannot supply evidence for the specified report. Reply 1 when both are credited or 0 otherwise."
)


def capture_notes(snapshots):
    notes = {}
    for directory in snapshots:
        manifest = json.loads((directory / "capture.json").read_bytes())
        parts = [(directory, manifest)] if "checksums" in manifest else [
            (directory / split, manifest[split]) for split in ("calibration", "test")
        ]
        for folder, part in parts:
            for doi, checksum in part["checksums"].items():
                raw = (folder / f"{doi.rsplit('.', 1)[-1]}.xml").read_bytes()
                if checksum != sha256(raw).hexdigest():
                    raise ValueError("Changed publisher capture")
                parsed = legacy.note(raw, doi)
                if parsed is not None:
                    notes[doi] = parsed
    return notes


def paired_fixture(snapshots):
    notes = capture_notes(snapshots)
    candidates = []
    for first, second in combinations(sorted(notes), 2):
        _, _, a, first_names = notes[first]
        _, _, b, second_names = notes[second]
        if not first_names.isdisjoint(second_names):
            continue
        for left, right in combinations(sorted(a.keys() & b.keys()), 2):
            both_a, both_b = a[left] & a[right], b[left] & b[right]
            flips = (both_a & (b[left] ^ b[right])) | (both_b & (a[left] ^ a[right]))
            for initials in sorted(flips):
                identity = (first, second, left, right, initials)
                candidates.append((legacy.stable(*identity), identity, initials in both_a))
    rows, used_dois, used_names = [], set(), set()
    for key, identity, first_supports in sorted(candidates):
        a, b, left, right, initials = identity
        names = notes[a][3] | notes[b][3]
        if a in used_dois or b in used_dois or not names.isdisjoint(used_names):
            continue
        used_dois.update((a, b))
        used_names.update(names)
        rows.append(dict(
            case_id=key.hex(), study_labels=["N17", "N54"], notes=[notes[a][1], notes[b][1]],
            initials=initials, tasks=[legacy.PHRASES[name][1] for name in (left, right)],
            positive_index=0 if first_supports else 1,
        ))
    return dict(source_pairs=len(rows), design_candidates=len(candidates),
                capture_sha256=[digest(f / "capture.json") for f in snapshots], rows=rows)


def run(fixture, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    data = json.loads(fixture.read_bytes())
    if data["source_pairs"] < 12:
        raise ValueError("Insufficient disjoint publisher pairs")
    judge = Judge()
    totals = {key: Counter() for key in ("full_room", "study_bound", "study_and_initials_bound", "pooled_union")}
    ms = {key: [] for key in totals}
    for index, case in enumerate(data["rows"]):
        initials = case["initials"]
        notes = case["notes"]
        labels = case["study_labels"]
        positive = case["positive_index"]
        observations = {method: [] for method in totals}
        order = (0, 1) if index % 2 else (1, 0)
        for requested in order:
            info = dict(report=labels[requested], initials=initials, required_tasks=case["tasks"])
            full = {labels[j]: notes[j] for j in order}
            baseline = {labels[requested]: notes[requested]}
            started = perf_counter()
            scoped = {labels[requested]: legacy.scope(notes[requested], initials)}
            scope_ms = (perf_counter() - started) * 1000
            for method, evidence in (("full_room", full), ("study_bound", baseline),
                                     ("study_and_initials_bound", scoped)):
                decision, timing = judge(TASK, dict(info, reports=evidence), (False, True))
                observations[method].append((requested == positive, decision))
                ms[method].append(timing + (scope_ms if method == "study_and_initials_bound" else 0.))
            observations["pooled_union"].append((requested == positive, True))
        for method, views in observations.items():
            totals[method].update(
                both_correct=int(all(expected == predicted for expected, predicted in views)),
                accepted_support=sum(expected and predicted for expected, predicted in views),
                wrong_study_acceptance=sum(not expected and predicted for expected, predicted in views),
                always_yes=int(all(predicted for _, predicted in views)),
            )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), cases=data["source_pairs"] * 2,
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        model_provenance_sha256=judge.sha, capture_sha256=data["capture_sha256"],
        source_pairs=data["source_pairs"], design_candidates=data["design_candidates"],
        results={arm: dict(values, p95_ms=float(np.quantile(ms[arm], .95)) if ms[arm] else None)
                 for arm, values in totals.items()},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("output", type=Path)
    make.add_argument("snapshots", nargs="+", type=Path)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("fixture", type=Path)
    evaluate.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "build":
        write_json(args.output, paired_fixture(args.snapshots))
    else:
        run(args.fixture, args.output)
