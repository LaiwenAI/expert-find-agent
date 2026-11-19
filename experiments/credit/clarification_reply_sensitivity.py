"""Measure exact expected outcomes under simulated clarification reply errors."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from experiments.credit import one_role_clarification as clarify


live = clarify.live
bundles = clarify.bundles
ERROR_RATES = (0.0, .05, .10, .20)
NONRESPONSE = (0.0, .20)
METHODS = ("trust_open", "trust_menu", "keep_rival", "confirm_independent", "confirm_shared")


def outcomes(case, encoder, ridge, cutoff, owners, cache, paths, labels, owner_masks):
    lines = case["request"].splitlines()[1:]
    scores = live.encode_tasks(encoder, [row[2:] for row in lines]) @ ridge.T
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    pools = [row[gaps[row] <= cutoff].tolist()
             for row, gaps in zip(order, scores.max(axis=1, keepdims=True) - scores, strict=True)]
    hypothesis = clarify.worlds(pools)
    status, matches, _ = clarify.lookup(hypothesis, owners, cache, paths)
    baseline = clarify.assess(status, matches, case, labels)
    slot = (clarify.choose("majority_gate", clarify.partitions(hypothesis), margins, owner_masks)
            if status == "unresolved" else None)
    if slot is None:
        return dict(id=case["id"], baseline=baseline, slot=None)
    gold = live.ROLES.index(case["roles"][slot])
    menu = sorted({roles[slot] for roles in hypothesis})
    replies = {}
    for answer in set(menu) | {gold}:
        rivals = [int(role) for role in order[slot] if role in menu and role != answer]
        replies[answer] = {}
        for arm, corrected in (("trust", [answer]), ("keep_rival", [answer, *rivals[:1]])):
            updated = pools[:]
            updated[slot] = corrected
            response, matches, _ = clarify.lookup(clarify.worlds(updated), owners, cache, paths)
            replies[answer][arm] = clarify.assess(response, matches, case, labels)
    return dict(id=case["id"], baseline=baseline, slot=slot, gold=gold, menu=menu,
                wrong=max((role for role in menu if role != gold), key=lambda role: scores[slot, role]),
                replies=replies)


def distribution(row, error, unavailable, channel, menu_only):
    if menu_only and row["gold"] not in row["menu"]:
        return {None: 1.0}
    replies = defaultdict(float)
    replies[None] = unavailable
    replies[row["gold"]] = (1 - unavailable) * (1 - error)
    rivals = [row["wrong"]] if channel == "top_rival" else [r for r in row["menu"] if r != row["gold"]]
    for rival in rivals:
        replies[rival] += (1 - unavailable) * error / len(rivals)
    return replies


def expected(rows, error, unavailable, channel, method):
    result = defaultdict(float)

    def count(item, weight):
        result["fully_supported_positive"] += weight * (item["positive"] and not item["unsafe"])
        result["unsafe_positive"] += weight * item["unsafe"]
        result["unresolved"] += weight * (item["status"] == "unresolved")
        result["correct_no_match"] += weight * item["correct_no_match"]
        result["false_no_match"] += weight * item["false_no_match"]

    for row in rows:
        original = row["baseline"]
        if row["slot"] is None:
            count(original, 1.)
            continue
        result["first_questions"] += 1
        answers = distribution(row, error, unavailable, channel, method == "trust_menu")
        if not math.isclose(sum(answers.values()), 1., abs_tol=1e-12):
            raise AssertionError("Reply distribution lost mass")
        for reply, chance in answers.items():
            if reply is None:
                count(original, chance)
                continue
            arm = "keep_rival" if method == "keep_rival" else "trust"
            outcome = row["replies"][reply][arm]
            if method.startswith("confirm") and outcome["status"] != "unresolved":
                result["follow_up_questions"] += chance
                repeated = chance if method == "confirm_independent" else .5 + .5 * chance
                count(outcome, chance * repeated)
                count(original, chance * (1 - repeated))
            else:
                count(outcome, chance)
    if not math.isclose(sum(result[key] for key in (
            "fully_supported_positive", "unsafe_positive", "unresolved", "correct_no_match", "false_no_match"
    )), len(rows), abs_tol=1e-10):
        raise AssertionError("Outcome distribution lost mass")
    return dict(sorted(result.items()))


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
    for person, docs in owners:
        mask = 0
        for doc in docs:
            mask |= labels[person, doc]
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
    ridge = banks["ridge"]
    scores = live.encode_tasks(encoder, [phrase for phrase, _ in calibration]) @ ridge.T
    cutoff = clarify.robust.cutoffs(scores, [live.ROLES.index(role) for _, role in calibration])[1.]
    with TemporaryDirectory(prefix="credit-reply-noise-", dir=scratch) as tmp:
        paths = {doc: Path(tmp) / f"{i}.json" for i, doc in enumerate(sorted({d for _, p in owners for d in p}))}
        for doc, path in paths.items():
            path.write_text(json.dumps({o: value for (o, source), value in labels.items() if source == doc}))
        rows = [outcomes(case, encoder, ridge, cutoff, owners, cache, paths, labels, owner_masks)
                for case in cases]
    results = {}
    for unavailable in NONRESPONSE:
        for error in ERROR_RATES:
            for channel in ("top_rival", "uniform"):
                results[f"{unavailable}/{error}/{channel}"] = {
                    arm: expected(rows, error, unavailable, channel, arm) for arm in METHODS
                }
    asks = [row for row in rows if row["slot"] is not None]
    bundles.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=live.torch.cuda.get_device_name(), requests=len(rows), asked=len(asks),
        gold_outside_menu=sum(row["gold"] not in row["menu"] for row in asks),
        fixture_sha256=bundles.digest(fixture), glossary_sha256=bundles.digest(glossary_file),
        training_sha256=bundles.digest(training), script_sha256=bundles.digest(Path(__file__)),
        selector_script_sha256=bundles.digest(Path(clarify.__file__)),
        embedding=embedding_info, fitted=fitted, cutoff=cutoff, results=results, rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "glossary_file", "training", "embedding", "scratch", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
