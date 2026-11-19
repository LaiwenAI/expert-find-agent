"""Test whether a Re-DocRED attribution gate transfers to publisher notes."""

import argparse
from collections import Counter
from itertools import combinations
import json
import math
import os
from pathlib import Path
from time import perf_counter

from experiments.credit import legacy_cascade_transfer as capture
from experiments.credit import legacy_contribution_text as legacy


RECEIPTS = (
    "f3a26e4350238038d4c0d835775d60475b630a53ea2cbf0e342014f99aac4f28",
    "01b02c2e4d718e343a623faa954622eadb35bedfef75fac2f6d8ff43a70f0fd2",
)
ROUTING_SHA = "c2bae05033a7733233ccd850ba1177787f5ba89e47cd0d52ab04f54367f41a53"
CUES = {
    "design": "conceived designed design",
    "execution": "performed conduct conducted experiment experimental execution",
    "analysis": "analyzed analysed analysis interpreted",
    "writing": "wrote writing drafted manuscript",
    "resources": "reagents materials resources supplied provided tools",
}


def digest(path):
    from hashlib import file_digest
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def write_json(path, content):
    with path.open("x") as output:
        json.dump(content, output)
        output.write("\n")


def choose_pair(doi, labels):
    all_names = set().union(*labels.values())
    pairs = [(role, author, wrong) for role, people in labels.items()
             for author in people for wrong in all_names - people]
    return min(pairs, key=lambda item: legacy.stable("single-role-transfer-v1", doi, *item), default=None)


def build(original, independent, output):
    paths = [folder / "capture.json" for folder in (original, independent)]
    if tuple(map(digest, paths)) != RECEIPTS:
        raise ValueError("Changed publisher capture receipts")
    old, new = [json.loads(path.read_bytes()) for path in paths]
    prior_dois = {doi for split in old.values() for doi in split["selected"]}
    if prior_dois & set(new["selected"]) or new["prior_sha256"] != RECEIPTS[0]:
        raise ValueError("Overlapping or mismatched publisher cohort")
    past_names = set()
    rows = {"dev": [], "test": []}
    name_sets = {"dev": [], "test": []}
    omissions = Counter()
    folders = [(split, original / split, info["checksums"]) for split, info in old.items()]
    folders.append(("transfer", independent, new["checksums"]))
    for split, folder, checksums in folders:
        for doi, expected in sorted(checksums.items()):
            raw = capture.checked_xml(folder, doi, expected)
            if split != "transfer":
                past_names.update(capture.byline_names(raw))
            if split == "test":
                continue
            panel = "test" if split == "transfer" else "dev"
            parsed = legacy.note(raw, doi)
            if parsed is None:
                omissions[f"{panel}_note"] += 1
                continue
            _, text, labels, names = parsed
            if panel == "test" and names & past_names:
                omissions["test_prior_author"] += 1
                continue
            sentences = legacy.SENTENCE.split(text)
            pair = choose_pair(doi, labels)
            if pair is None or len(sentences) < 2:
                omissions[f"{panel}_contrast"] += 1
                continue
            role, supported, rival = pair
            rows[panel].append(dict(id=doi, role=role, people=[supported, rival], sentences=sentences))
            name_sets[panel].append(names)
    for split, cases in rows.items():
        for case, group in zip(cases, capture.clusters(name_sets[split]), strict=True):
            case["group"] = group
    write_json(output, dict(receipts=RECEIPTS, omissions=dict(omissions), panels=rows))


def window(case, base):
    words, _, idf = base.profile(case["sentences"])
    names = [tuple(base.tokens(name)) for name in case["people"]]
    cues = tuple(base.tokens(CUES[case["role"]]))
    name_scores = [[max(float(base.contains(row, name)), .6 * base.coverage(name, row, idf))
                    for row in words] for name in names]
    cue_scores = [base.coverage(cues, row, idf) for row in words]

    def key(pair):
        a, b = pair
        people = [max(scores[a], scores[b]) for scores in name_scores]
        return sum(people) + max(cue_scores[a], cue_scores[b]), min(people), -a, -b

    return max(combinations(range(len(words)), 2), key=key)


def proximity(case, indices, base):
    names = [tuple(base.tokens(name)) for name in case["people"]]
    cues = set(base.tokens(CUES[case["role"]]))
    best = [0., 0.]
    for i in indices:
        words = tuple(base.tokens(case["sentences"][i]))
        cue_positions = [j for j, word in enumerate(words) if word in cues]
        for p, name in enumerate(names):
            starts = [k for k in range(len(words)-len(name)+1) if words[k:k+len(name)] == name]
            distance = min((min(abs(k-j), abs(k+len(name)-1-j)) for k in starts for j in cue_positions),
                           default=math.inf)
            best[p] = max(best[p], 1. / (1 + distance))
    return tuple(best)


def reader_scores(case, indices, score, base):
    order = 1 - 2 * (int(base.stable(case["id"])[0], 16) % 2)
    source = "\n".join(case["sentences"][i] for i in indices)
    requirement = legacy.PHRASES[case["role"]][1]
    prompts = [(f"Does the contribution note explicitly credit author {name} with this task? {requirement}",
                source) for name in case["people"][::order]]
    values, truncated = score(prompts)
    if truncated or len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid paired verifier output")
    return values[::order]


def evaluate(cases, arms, score, base, route):
    details = {}
    for i, case in enumerate(cases):
        names = tuple(arms)
        offset = i % len(names)
        for arm in names[offset:] + names[:offset]:
            settings = arms[arm]
            start = perf_counter()
            indices = tuple(range(len(case["sentences"]))) if arm == "local_full" else window(case, base)
            call = True
            if "low" in settings:
                values = proximity(case, indices, base)
                call = bool(route.fallback(route.np.asarray(values), settings))
            if call:
                values = reader_scores(case, indices, score, base)
            cutoff = (settings["reader_cutoff"] if call else settings["high"]) if "low" in settings else settings["cutoff"]
            verdict = [value > cutoff for value in values]
            elapsed = (perf_counter() - start) * 1000
            details.setdefault(arm, []).append((verdict, call, elapsed))
    return details


def run(panel, prior, reader, output):
    import numpy as np
    import torch
    from expert_search.statistics import clustered_paired_interval
    from experiments.docred import oracle_role_evidence as oracle
    from experiments.docred import person_bound_evidence as base
    from experiments.docred import route_role_verification as route

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    if digest(prior) != ROUTING_SHA:
        raise ValueError("Changed Re-DocRED routing result")
    prior_results = json.loads(prior.read_bytes())
    dataset = json.loads(panel.read_bytes())
    if tuple(dataset["receipts"]) != RECEIPTS:
        raise ValueError("Changed source lineage")
    tokenizer, reader_model, provenance = oracle.load_reader(reader)
    if provenance["revision"] != prior_results["checkpoints"]["reader"]:
        raise ValueError("Different reader checkpoint")
    score = oracle.reader_scorer(tokenizer, reader_model)
    development = dataset["panels"]["dev"]
    cap = len(development) // 20
    cheap, short, full = [], [], []
    for case in development:
        indices = window(case, base)
        cheap.append(proximity(case, indices, base))
        short.append(reader_scores(case, indices, score, base))
        full.append(reader_scores(case, tuple(range(len(case["sentences"]))), score, base))
    frozen = prior_results["calibration"]["settings"]
    short_cutoff = sorted((rival for _, rival in short), reverse=True)[cap]
    arms = dict(
        frozen_reader=frozen["reader"],
        frozen_gate=frozen["lexical_75"],
        frozen_band_local_cutoff={**frozen["lexical_75"], "reader_cutoff": short_cutoff},
        local_reader=dict(cutoff=short_cutoff),
        local_gate=route.calibrate(np.asarray(cheap), np.asarray(short), int(.75 * len(development)), cap),
        local_full=dict(cutoff=sorted((rival for _, rival in full), reverse=True)[cap]),
    )
    results = {}
    for split, cases in dataset["panels"].items():
        live = evaluate(cases, arms, score, base, route)
        groups = [case["group"] for case in cases]
        report = {}
        reference = [int(a and not b) for (a, b), _, _ in live["local_reader"]]
        for method, observations in live.items():
            verdicts, calls, times = zip(*observations, strict=True)
            both = [int(a and not b) for a, b in verdicts]
            report[method] = dict(
                accepted_supported=sum(a for a, _ in verdicts), accepted_rival=sum(b for _, b in verdicts),
                accepted_rival_fast=sum(rival and not call for (_, rival), call in zip(verdicts, calls, strict=True)),
                both_correct=sum(both), reader_calls=sum(calls),
                mean_ms=float(np.mean(times)), p95_ms=float(np.quantile(times, .95)),
                both_delta_vs_local_reader=clustered_paired_interval(np.subtract(both, reference), groups),
            )
        if split == "dev":
            assert all(report[method]["accepted_rival"] <= cap for method in ("local_reader", "local_gate", "local_full"))
        results[split] = dict(pairs=len(cases), groups=len(set(groups)), arms=report)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=provenance["revision"],
        code_sha256=digest(Path(__file__)), projection_sha256=digest(panel), upstream_routing_sha256=ROUTING_SHA,
        capture_sha256=RECEIPTS, omissions=dataset["omissions"], local_development_rival_cap=cap,
        settings=arms, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest="command", required=True)
    build_args = choices.add_parser("build")
    for argument in ("original", "independent", "output"):
        build_args.add_argument(argument, type=Path)
    run_args = choices.add_parser("run")
    for argument in ("panel", "prior", "reader", "output"):
        run_args.add_argument(argument, type=Path)
    args = vars(parser.parse_args())
    {"build": build, "run": run}[args.pop("command")](**args)
