"""Test conservative recommendations under multiple role interpretations."""

import argparse
import json
import os
from itertools import product
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from experiments.credit import live_expert_bundles as live


bundles = live.bundles
MODES = ("point", "abstain80", "robust80", "robust90", "robust100", "top2")
COVERAGE = {"abstain80": .80, "robust80": .80, "robust90": .90, "robust100": 1.0}


def cutoffs(scores, roles):
    residual = scores.max(axis=1) - scores[np.arange(len(scores)), roles]
    return {fraction: float(np.sort(residual)[int(np.ceil(fraction * len(residual))) - 1])
            for fraction in sorted(set(COVERAGE.values()))}


def assignments(order, gap, mode, thresholds):
    if mode == "point":
        pool = [row[:1] for row in order]
    elif mode == "top2":
        pool = [row[:2] for row in order]
    else:
        ceiling = thresholds[COVERAGE[mode]]
        pool = [row[row_gaps[row] <= ceiling] for row, row_gaps in zip(order, gap, strict=True)]
    return {sum(1 << int(i) for i in triple) for triple in product(*pool) if len(set(triple)) == len(pool)}


def query(mode, case, encoder, ridge, thresholds, owners, cached, paths, truth):
    started = perf_counter()
    header, *items = case["request"].splitlines()
    if header != live.HEADER or len(items) != 3 or any(not item.startswith("- ") for item in items):
        raise ValueError("Expected the pinned request format")
    scores = live.encode_tasks(encoder, [item[2:] for item in items]) @ ridge.T
    order = np.argsort(-scores, axis=1, kind="stable")
    gap = scores.max(axis=1, keepdims=True) - scores
    hypotheses = assignments(order, gap, mode, thresholds)
    union = 0
    for hypothesis in hypotheses:
        union |= hypothesis
    matches = []
    disk = {}

    def read(owner, source):
        if source not in disk:
            disk[source] = json.loads(paths[source].read_bytes())
        return disk[source][owner]

    if hypotheses and (mode != "abstain80" or len(hypotheses) == 1):
        matches, _, _ = bundles.rank(owners, cached, union, read, "frontier")
    status = "supported" if matches else "no_match" if len(hypotheses) == 1 else "unresolved"
    response = json.dumps(dict(status=status, matches=matches), ensure_ascii=True, separators=(",", ":"))
    elapsed = (perf_counter() - started) * 1000
    unsupported = 0
    for owner, sources in matches:
        observed = 0
        for source in sources:
            observed |= truth[owner, source]
        unsupported += observed & case["mask"] != case["mask"]
    contains = case["mask"] in hypotheses
    if contains and unsupported:
        raise AssertionError("A returned expert failed an included interpretation")
    return dict(
        response=response, ms=elapsed, hypotheses=len(hypotheses), contains_intended_roles=contains,
        status=status, positive=len(matches), unsupported=unsupported, files=len(disk),
        exact_response=status != "unresolved" and matches == case["expected"],
    )


def summarize(rows):
    output = {}
    for mode in MODES:
        observed = [row["arms"][mode][0] for row in rows]
        output[mode] = dict(
            requests=len(rows), contains=sum(item["contains_intended_roles"] for item in observed),
            positive_requests=sum(item["positive"] > 0 for item in observed),
            correct_positive=sum(item["positive"] > 0 and item["unsupported"] == 0 for item in observed),
            returned=sum(item["positive"] for item in observed),
            unsupported_recommendations=sum(item["unsupported"] for item in observed),
            unsupported_requests=sum(item["unsupported"] > 0 for item in observed),
            unresolved=sum(item["status"] == "unresolved" for item in observed),
            false_no_match=sum(bool(row["expected_positive"]) and item["status"] == "no_match"
                               for row, item in zip(rows, observed, strict=True)),
            exact_responses=sum(item["exact_response"] for item in observed),
            hypotheses_median=float(np.median([item["hypotheses"] for item in observed])),
            p95_ms=float(np.quantile([median(v["ms"] for v in row["arms"][mode]) for row in rows], .95)),
        )
    return output


def run(fixture, glossary_file, training, embedding, scratch, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not live.torch.cuda.is_available():
        raise RuntimeError("Require a new output and Slurm CUDA")
    if bundles.digest(fixture) != bundles.FIXTURE_SHA256 or bundles.digest(glossary_file) != live.NISO_SHA256:
        raise ValueError("Changed benchmark input")
    sources = json.loads(fixture.read_bytes())["sources"]
    glossary = json.loads(glossary_file.read_bytes())
    owners, records = bundles.owner_data(sources)
    owners = [(owner, docs) for owner, docs in owners if len(docs) > 1]
    bits = {role: 1 << i for i, role in enumerate(live.ROLES)}
    truth = {witness: bundles.role_mask(roles, bits) for witness, roles in records.items()}
    caps = {row["doi"]: bundles.role_mask({r for a in row["authors"] for r in a["roles"]}, bits) for row in sources}
    cached = bundles.cached(owners, truth, caps, live.CACHE)
    calibration, held = live.phrase_split(glossary)
    cases = live.panel(owners, truth, bits, held)
    encoder_info = live.verify_checkpoint(embedding)
    if encoder_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = live.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    definitions = encoder.encode([glossary["definitions"][role] for role in live.ROLES])
    banks, fitted = live.train_banks(training, SimpleNamespace(embedder=encoder, vectors=definitions),
                                    glossary["examples"])
    ridge = banks["ridge"]
    calibration_scores = live.encode_tasks(encoder, [p for p, _ in calibration]) @ ridge.T
    thresholds = cutoffs(calibration_scores, [live.ROLES.index(role) for _, role in calibration])
    rows = []
    with TemporaryDirectory(prefix="credit-robust-bundles-", dir=scratch) as temp:
        paths = {doi: Path(temp) / f"{i}.json" for i, doi in enumerate(sorted({d for _, p in owners for d in p}))}
        for doi, path in paths.items():
            path.write_text(json.dumps({o: value for (o, source), value in truth.items() if source == doi}))
        for mode in MODES:
            query(mode, cases[0], encoder, ridge, thresholds, owners, cached, paths, truth)
        for i, case in enumerate(cases):
            result = {mode: [] for mode in MODES}
            for repeat in range(2):
                shift = (i + repeat) % len(MODES)
                for mode in MODES[shift:] + MODES[:shift]:
                    result[mode].append(query(mode, case, encoder, ridge, thresholds, owners, cached, paths, truth))
            if any(items[0]["response"] != items[1]["response"] for items in result.values()):
                raise AssertionError("Repeated run returned different output")
            rows.append(dict(id=case["id"], cohort=case["cohort"], expected_positive=bool(case["expected"]),
                             arms=result))
    bundles.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=live.torch.cuda.get_device_name(), calibration_phrases=len(calibration),
        script_sha256=bundles.digest(Path(__file__)), router_sha256=bundles.digest(Path(live.__file__)),
        fixture_sha256=bundles.digest(fixture), glossary_sha256=bundles.digest(glossary_file),
        training_sha256=bundles.digest(training), bundle_script_sha256=bundles.digest(Path(bundles.__file__)),
        embedding=encoder_info, training=fitted, cutoffs=thresholds, report=summarize(rows),
        by_stratum={name: summarize([row for row in rows if row["cohort"] == name])
                    for name in sorted({row["cohort"] for row in rows})}, rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("fixture", "glossary_file", "training", "embedding", "scratch", "output"):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
