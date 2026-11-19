"""Test whether exact section headings disambiguate identical requirement excerpts."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_requirement_roles import ROLES, ScopedReader, prompt_data, spans


ARMS = ("full", "local", "heading")


def heading_offsets(text):
    result = []
    start = 0
    for line in text.splitlines(keepends=True):
        clean = line.strip()
        if 3 <= len(clean) <= 96 and clean.endswith((":", "\uff1a")):
            left = start + len(line) - len(line.lstrip())
            result.append((left, left + len(clean)))
        start += len(line)
    return result


def scoped_context(text, clauses):
    headings = heading_offsets(text)
    ends = [end for _, end in headings]
    used = {}
    for i, clause in enumerate(clauses):
        j = bisect_right(ends, clause["start"]) - 1
        if j < 0:
            continue
        heading = headings[j]
        if heading[0] < clauses[max(0, i - 1)]["start"]:
            used[i] = heading

    def context(index):
        parts = prompt_data(text, clauses, index, False)
        if index in used:
            a, b = used[index]
            # Keep the source label before the local evidence. The target remains untouched.
            return dict(section_heading=text[a:b], **parts)
        return parts

    return context, used


def evaluate(reader, row, offset):
    text = row["text"]
    clauses = spans(text)
    target = row.get("target")
    indices = [i for i, span in enumerate(clauses) if text[span["start"]:span["end"]] == target] if target else range(len(clauses))
    if target and len(indices) != 1:
        raise ValueError("Control needs one exact target span")
    context, added = scoped_context(text, clauses)
    if target and row["layout"] != "adjacent" and indices[0] not in added:
        raise ValueError("Non-adjacent controlled heading was not injected")
    reports = {}
    rotation = offset % len(ARMS)
    for method in ARMS[rotation:] + ARMS[:rotation]:
        start = perf_counter()
        labels, slots = reader.judge(text, clauses, indices, method == "full",
                                     context=context if method == "heading" else None)
        reports[method] = dict(roles={str(i): ROLES[label] for i, (label, _) in labels.items()},
                               padded_tokens=slots, ms=(perf_counter()-start)*1000)
    return dict(id=row["id"], pair=row.get("pair"), language=row.get("language"),
                layout=row.get("layout"), gold=row.get("role"),
                injected_headings=sum(i in added for i in indices), arms=reports)


def summarize(rows):
    paired = defaultdict(list)
    for row in rows:
        if row["pair"] is not None:
            paired[row["pair"]].append(row)
    if any(len(group) != 2 for group in paired.values()):
        raise ValueError("Each paired heading intervention needs exactly two cases")
    results = {}
    for name in ARMS:
        per_arm = [row["arms"][name] for row in rows]
        counts = Counter()
        by_layout = {}
        for row, arm in zip(rows, per_arm, strict=True):
            for index, role in arm["roles"].items():
                counts["classified"] += 1
                counts["prior_qualification_views"] += role == ROLES[2]
                full = row["arms"]["full"]["roles"][index]
                counts["agrees_with_full"] += role == full
                counts["shares_prior_status"] += (role == ROLES[2]) == (full == ROLES[2])
                if row["gold"] is not None:
                    counts["correct"] += role == row["gold"]
                    counts["false_prior_qualification"] += role == ROLES[2] and row["gold"] != role
                    counts["missed_prior_qualification"] += row["gold"] == ROLES[2] and role != row["gold"]
                    by_layout.setdefault(row["layout"], Counter()).update(
                        cases=1, correct=role == row["gold"])
        results[name] = dict(
            **counts, both_pair_roles_correct=sum(
                all(next(iter(row["arms"][name]["roles"].values())) == row["gold"] for row in group)
                for group in paired.values()
            ), by_layout={k:dict(v) for k,v in by_layout.items()},
            padded_tokens=sum(arm["padded_tokens"] for arm in per_arm),
            p95_ms=float(np.quantile([arm["ms"] for arm in per_arm], .95)),
        )
    return dict(requests=len(rows), pairs=len(paired), injected_headings=sum(row["injected_headings"] for row in rows),
                arms=results)


def run(development, controls, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm allocation is required")
    dev = [json.loads(line) for line in development.read_bytes().splitlines()]
    if len(dev) != 80 or any(row["split"] != "development" for row in dev):
        raise ValueError("Use only the documented 80-request development export")
    synthetic = json.loads(controls.read_bytes())
    groups = defaultdict(list)
    for row in synthetic:
        groups[row["pair"]].append((row["target"], row["role"]))
    if any(len(group) != 2 or group[0][0] != group[1][0] or {r for _,r in group} != set(ROLES[1:3])
           for group in groups.values()):
        raise ValueError("Each pair must keep the target text and change its role")
    reader = ScopedReader(checkpoint)
    if not reader.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    evaluate(reader, synthetic[0], 0)
    report, traces = {}, {}
    for name, dataset in (("controls", synthetic), ("development", dev)):
        rows = []
        for i, row in enumerate(dataset):
            rows.append(evaluate(reader, row, i))
            if i % 20 == 19:
                print(name, i+1, "complete", flush=True)
        report[name] = summarize(rows)
        traces[name] = rows
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=reader.torch.cuda.get_device_name(), model=reader.provenance,
        script_sha256=digest(Path(__file__)), role_reader_sha256=digest(
            Path(__file__).with_name("enterprise_requirement_roles.py")
        ), controls_sha256=digest(controls), development_sha256=digest(development),
        report=report, traces=traces,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
