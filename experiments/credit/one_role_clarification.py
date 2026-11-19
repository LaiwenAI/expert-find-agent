"""Replay one exact clarification on unresolved expert-search requests."""

import argparse
from collections import defaultdict
from itertools import product
import json
import math
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from experiments.credit import robust_role_bundles as robust


live = robust.live
bundles = robust.bundles
MODES = ("none", "first", "lowest_margin", "entropy", "evidence", "evidence_gate", "majority_gate")


def worlds(pools):
    return [roles for roles in product(*pools) if len(set(roles)) == len(pools)]


def role_union(hypotheses):
    mask = 0
    for roles in hypotheses:
        for role in roles:
            mask |= 1 << int(role)
    return mask


def partitions(hypotheses):
    groups = []
    for i in range(3):
        roles = defaultdict(list)
        for hypothesis in hypotheses:
            roles[hypothesis[i]].append(hypothesis)
        groups.append(roles)
    return groups


def branch_value(group, owner_masks):
    total = sum(map(len, group.values()))
    expected = positive = 0
    entropy = 0.0
    for branch in group.values():
        weight = len(branch) / total
        entropy -= weight * math.log(weight)
        mask = role_union(branch)
        count = sum(roles & mask == mask for roles in owner_masks)
        expected += len(branch) * min(bundles.TOP, count)
        if count:
            positive += len(branch)
    return expected / total, positive / total, entropy


def choose(mode, groups, margins, owner_masks):
    active = [i for i, group in enumerate(groups) if len(group) > 1]
    if mode == "none" or not active:
        return None
    if mode == "first":
        return active[0]
    if mode == "lowest_margin":
        return min(active, key=lambda i: (margins[i], i))

    if mode == "entropy":
        def entropy(i):
            total = sum(map(len, groups[i].values()))
            weights = [len(branch) / total for branch in groups[i].values()]
            return -sum(weight * math.log(weight) for weight in weights)

        return max(active, key=lambda i: (entropy(i), -i))

    values = {i: branch_value(groups[i], owner_masks) for i in active}
    if mode == "evidence_gate":
        active = [i for i in active if values[i][1] > 0]
    if mode == "majority_gate":
        active = [i for i in active if values[i][1] >= .5]
    return max(active, key=lambda i: (values[i][0], values[i][2], -i), default=None)


def lookup(hypotheses, owners, cache, paths):
    if not hypotheses:
        return "unresolved", [], 0
    parsed = {}

    def read(owner, source):
        if source not in parsed:
            parsed[source] = json.loads(paths[source].read_bytes())
        return parsed[source][owner]

    matches, _, _ = bundles.rank(owners, cache, role_union(hypotheses), read, "frontier")
    unique = len({frozenset(roles) for roles in hypotheses})
    status = "supported" if matches else "no_match" if unique == 1 else "unresolved"
    return status, matches, len(parsed)


def assess(status, matches, case, labels):
    unsupported = 0
    for owner, documents in matches:
        observed = 0
        for doc in documents:
            observed |= labels[owner, doc]
        unsupported += observed & case["mask"] != case["mask"]
    return dict(status=status, positive=bool(matches), unsafe=bool(unsupported),
                false_no_match=bool(case["expected"]) and status == "no_match",
                correct_no_match=not case["expected"] and status == "no_match")


def test_case(case, encoder, bank, cutoff, owners, cache, paths, labels, owner_masks):
    start = perf_counter()
    title, *lines = case["request"].splitlines()
    if title != live.HEADER or len(lines) != 3 or any(not line.startswith("- ") for line in lines):
        raise ValueError("Expected the pinned request format")
    scores = live.encode_tasks(encoder, [line[2:] for line in lines]) @ bank.T
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    pools = [np.flatnonzero(row.max() - row <= cutoff).tolist() for row in scores]
    initial_worlds = worlds(pools)
    status, initial, files = lookup(initial_worlds, owners, cache, paths)
    groups = partitions(initial_worlds) if status == "unresolved" else None
    shared_ms = (perf_counter() - start) * 1000
    answers = [live.ROLES.index(role) for role in case["roles"]]

    def clarified(index):
        fixed = pools[:]
        fixed[index] = [answers[index]]
        hypotheses = worlds(fixed)
        after, matches, opened = lookup(hypotheses, owners, cache, paths)
        return after, matches, opened

    result = {}
    for mode in MODES:
        start = perf_counter()
        slot = choose(mode, groups, margins, owner_masks) if groups is not None else None
        current, matches, opened = (status, initial, 0) if slot is None else clarified(slot)
        json.dumps(dict(status=current, matches=matches))
        ms = shared_ms + (perf_counter() - start) * 1000
        result[mode] = dict(**assess(current, matches, case, labels), slot=slot, ms=ms, files=files+opened)
    feasible = []
    if groups is not None:
        for slot, options in enumerate(groups):
            if len(options) > 1:
                current, matches, _ = clarified(slot)
                feasible.append(assess(current, matches, case, labels))
    return dict(id=case["id"], cohort=case["cohort"], methods=result,
                oracle_can_recover_supported=any(row["positive"] and not row["unsafe"] for row in feasible),
                oracle_can_recover_correct_no_match=any(row["correct_no_match"] for row in feasible))


def summarize(rows):
    output = {}
    for mode in MODES:
        outcomes = [row["methods"][mode] for row in rows]
        output[mode] = dict(
            questions=sum(row["slot"] is not None for row in outcomes),
            positive=sum(row["positive"] for row in outcomes),
            supported=sum(row["positive"] and not row["unsafe"] for row in outcomes),
            unsupported=sum(row["unsafe"] for row in outcomes),
            unresolved=sum(row["status"] == "unresolved" for row in outcomes),
            correct_no_match=sum(row["correct_no_match"] for row in outcomes),
            false_no_match=sum(row["false_no_match"] for row in outcomes),
            p95_machine_ms=float(np.quantile([row["ms"] for row in outcomes], .95)),
            mean_files=mean(row["files"] for row in outcomes),
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
    labels = {key: bundles.role_mask(roles, bits) for key, roles in records.items()}
    caps = {source["doi"]: bundles.role_mask({r for a in source["authors"] for r in a["roles"]}, bits)
            for source in sources}
    owner_masks = []
    for owner, documents in owners:
        mask = 0
        for doc in documents:
            mask |= labels[owner, doc]
        owner_masks.append(mask)
    cache = bundles.cached(owners, labels, caps, live.CACHE)
    calibration, held = live.phrase_split(glossary)
    cases = live.panel(owners, labels, bits, held)
    embedding_info = live.verify_checkpoint(embedding)
    if embedding_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = live.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    definitions = encoder.encode([glossary["definitions"][role] for role in live.ROLES])
    banks, fitted = live.train_banks(training, SimpleNamespace(embedder=encoder, vectors=definitions),
                                    glossary["examples"])
    bank = banks["ridge"]
    calibration_scores = live.encode_tasks(encoder, [phrase for phrase, _ in calibration]) @ bank.T
    cutoff = robust.cutoffs(calibration_scores, [live.ROLES.index(role) for _, role in calibration])[1.0]
    with TemporaryDirectory(prefix="credit-clarification-", dir=scratch) as tmp:
        paths = {doc: Path(tmp) / f"{i}.json" for i, doc in enumerate(sorted({d for _, p in owners for d in p}))}
        for doc, path in paths.items():
            path.write_text(json.dumps({o: value for (o, source), value in labels.items() if source == doc}))
        test_case(cases[0], encoder, bank, cutoff, owners, cache, paths, labels, owner_masks)
        rows = [test_case(case, encoder, bank, cutoff, owners, cache, paths, labels, owner_masks) for case in cases]
    bundles.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=live.torch.cuda.get_device_name(), queries=len(rows), cutoff=cutoff,
        script_sha256=bundles.digest(Path(__file__)), robust_script_sha256=bundles.digest(Path(robust.__file__)),
        embedding=embedding_info, fitted=fitted, fixture_sha256=bundles.digest(fixture),
        glossary_sha256=bundles.digest(glossary_file), training_sha256=bundles.digest(training),
        report=summarize(rows),
        oracle_increment={field: sum(row[f"oracle_can_recover_{field}"] for row in rows)
                          for field in ("supported", "correct_no_match")}, rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "glossary_file", "training", "embedding", "scratch", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
