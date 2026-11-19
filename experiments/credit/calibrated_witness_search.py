"""Compare live same-owner witness schedules at a common declaration risk target."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import os
from time import perf_counter

import numpy as np

from experiments.credit import linked_witness_dispatch as replay
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold


BUDGETS = (2, 4)
CAPS = (10, 20, 30)
ARMS = ("round_robin", "owner_mean", "adaptive", "owner_mean_batch")


def split_components(items):
    members = {}
    for left, right in sorted({row[0] for row in items}):
        group = members.get(left, {left}) | members.get(right, {right})
        for doi in group:
            members[doi] = group
    components = sorted({tuple(sorted(group)) for group in members.values()}, key=lambda c: replay.stable(c))
    if len(components) < 4:
        raise ValueError("Need separated source components")
    calibration = {doi for group in components[::3] for doi in group}
    return ([row for row in items if row[0][0] in calibration],
            [row for row in items if row[0][0] not in calibration], components)


def format_pair(phrase, declaration):
    return dict(source=f"The complete statement credits the researcher with: {declaration}",
                criterion=f"The researcher personally performed the following contribution: {phrase}")


def panels(documents, glossary, encoder):
    items, links = replay.panel(documents)
    cal, test, components = split_components(items)
    phrases = defaultdict(set)
    for entry in glossary["examples"]:
        phrases[entry["role"]].add(entry["text"])
    indexed = {}
    for dois in sorted({row[0] for row in items}):
        sides = [{r["owner"]: set(r["roles"]) for r in documents[doi]["owners"]} for doi in dois]
        common = sorted(sides[0].keys() & sides[1].keys())
        rows = [(side, owner, sides[side][owner]) for owner in common for side in (0, 1)]
        texts = [" ".join(glossary["definitions"][role] for role in sorted(roles)) for _, _, roles in rows]
        indexed[dois] = rows, texts
    cache = sorted({text for _, texts in indexed.values() for text in texts})
    vectors = dict(zip(cache, encoder.encode(cache), strict=True))
    return cal, test, {
        dois: (rows, texts, np.stack([vectors[t] for t in texts]))
        for dois, (rows, texts) in indexed.items()
    }, phrases, components, links


def request_words(item, phrases):
    _, family, _, requested, _ = item
    return [min(phrases[role], key=lambda phrase: replay.stable(family, role, phrase)) for _, role in requested]


def format_reads(indices, rows, texts, words, requested):
    return [(i, role, format_pair(words[j], texts[i]))
            for i in indices for j, (side, role) in enumerate(requested) if side == rows[i][0]]


def read_checks(checks, score, batch_size):
    estimates = score([pair for _, _, pair in checks], batch_size=batch_size)
    if any(value is None for value in estimates):
        raise ValueError("Unexpected over-limit witness")
    return [(i, role, value[1]) for (i, role, _), value in zip(checks, estimates, strict=True)]


def round_robin(rows, scores, scopes):
    scoped = np.where([[side == target for target in scopes] for side, _, _ in rows], scores, -np.inf)
    rankings = [sorted((i for i, (side, _, _) in enumerate(rows) if side == slot),
                       key=lambda i: (-float(scoped[i].max()), i)) for slot in (0, 1)]
    return [i for pair in zip(*rankings, strict=True) for i in pair]


def search(rows, scores, texts, words, requested, arm, budget, score, threshold):
    scopes, roles = tuple(zip(*requested, strict=True))
    observed = [(side, owner, set()) for side, owner, _ in rows]
    opened, output_checks = [], []
    if arm == "round_robin":
        fixed = round_robin(rows, scores, scopes)[:budget]
    elif arm.startswith("owner_mean"):
        fixed = []
        for _ in range(budget):
            choice = replay.choose_read(scores, observed, scopes, roles, "owner_mean", fixed)
            if choice is None:
                break
            fixed.append(choice)
    else:
        fixed = None
    while len(opened) < budget:
        if fixed is None:
            choice = replay.choose_read(scores, observed, scopes, roles, "adaptive", opened)
            chosen = [] if choice is None else [choice]
        elif arm == "owner_mean_batch":
            chosen = fixed
        else:
            chosen = fixed[len(opened):len(opened)+1]
        if not chosen:
            break
        checks = format_reads(chosen, rows, texts, words, requested)
        estimates = read_checks(checks, score, len(checks))
        for i, role, margin in estimates:
            if margin > threshold:
                observed[i][2].add(role)
        output_checks.extend(estimates)
        opened.extend(chosen)
        if arm == "owner_mean_batch":
            break
    return replay.output(observed, opened, requested), opened, output_checks


def calibration_counts(cal, indexed, phrases, score):
    margins = {}
    for item in cal:
        dois, _, _, requested, _ = item
        rows, texts, _ = indexed[dois]
        words = request_words(item, phrases)
        checks = format_reads(range(len(rows)), rows, texts, words, requested)
        estimates = read_checks(checks, score, 1)
        for (idx, role, pair), (_, _, margin) in zip(checks, estimates, strict=True):
            key = dois, idx, role, pair["criterion"]
            margins[key] = role in rows[idx][2], margin
    flags, values = map(np.asarray, zip(*margins.values(), strict=True))
    return {cap: dict(threshold=threshold, support=int(flags.sum()), wrong_role=int((~flags).sum()),
                      false_accept=int(np.sum((values > threshold) & ~flags)),
                      true_accept=int(np.sum((values > threshold) & flags)))
            for cap in CAPS for threshold in [operating_threshold(values[~flags], cap)]}


def run(projection, glossary_file, embedding, checkpoint, output):
    import torch
    from experiments.scifact.small_nli_verifier import load, measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if replay.digest(projection) != replay.PROJECTION or replay.digest(glossary_file) != replay.EXAMPLES:
        raise ValueError("Unexpected benchmark input")
    documents = json.loads(projection.read_text())["documents"]
    glossary = json.loads(glossary_file.read_text())
    embedding_info = replay.verify_checkpoint(embedding)
    encoder = replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    cal, test, indexed, phrases, components, links = panels(documents, glossary, encoder)
    tokenizer, reader, positions, reader_info = load(checkpoint, "minilm")

    def score(pairs, *, batch_size):
        return measure(tokenizer, reader, positions, pairs, batch_size=batch_size)[0]

    calibration = calibration_counts(cal, indexed, phrases, score)
    exact_fields = defaultdict(Counter)
    counts, times, grouped, decisions = (defaultdict(factory) for factory in (Counter, list, lambda: defaultdict(list), list))
    for index, item in enumerate([test[0], *test]):
        dois, family, variant, requested, truth = item
        rows, texts, vectors = indexed[dois]
        words = request_words(item, phrases)
        start = perf_counter()
        scores = vectors @ encoder.encode(words, query=True).T
        prefix = (perf_counter() - start) * 1000
        arms = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for budget in BUDGETS:
            for arm in arms:
                for cap in CAPS:
                    threshold = calibration[cap]["threshold"]
                    start = perf_counter()
                    found, opened, checks = search(rows, scores, texts, words, requested, arm, budget, score, threshold)
                    elapsed = prefix + (perf_counter() - start) * 1000
                    if not index:
                        continue
                    key = cap, budget, arm
                    c = counts[key]
                    c.update(dict(queries=1, opens=len(opened), checks=len(checks), positive_queries=bool(truth),
                                  true_positive_queries=bool(found & truth), false_positive_queries=bool(found - truth),
                                  negative_promotions=bool(found) and not truth, exact=found == truth,
                                  false_atomic_support=sum(margin > threshold and role not in rows[i][2]
                                                           for i, role, margin in checks)))
                    times[key].append(elapsed)
                    grouped[key][family].append(found == truth)
                    decisions[key].append((found, opened))
                if index and arm != "owner_mean_batch":
                    if arm == "round_robin":
                        exact_reads = round_robin(rows, scores, [side for side, _ in requested])[:budget]
                        exact = replay.output(rows, exact_reads, requested)
                    else:
                        exact, exact_reads = replay.request(rows, scores, requested, budget, arm)
                    exact_fields[budget, arm].update(dict(queries=1, recovered_positive=bool(exact & truth),
                                                          exact=exact == truth, reads=len(exact_reads)))
    report = {}
    for cap in CAPS:
        report[cap] = {}
        for budget in BUDGETS:
            methods = {}
            batch_matches = sum(a == b for a, b in zip(decisions[cap, budget, "owner_mean"],
                                                       decisions[cap, budget, "owner_mean_batch"], strict=True))
            for arm in ARMS:
                key = cap, budget, arm
                methods[arm] = dict(counts[key], both_variants_exact=sum(len(v) == 2 and all(v) for v in grouped[key].values()),
                                    p95_ms=float(np.quantile(times[key], .95)))
            report[cap][budget] = dict(batch_matches_serial=batch_matches, arms=methods)
    replay.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                    script_sha256=replay.digest(Path(__file__)), dispatch_sha256=replay.digest(Path(replay.__file__)),
                                    projection_sha256=replay.PROJECTION, glossary_sha256=replay.EXAMPLES,
                                    embedding_revision=embedding_info["revision"], reader_revision=reader_info["revision"],
                                    calibration=calibration, exact_fields={budget: {arm: exact_fields[budget, arm] for arm in ARMS
                                        if arm != "owner_mean_batch"} for budget in BUDGETS}, split=dict(
                                        calibration_requests=len(cal), test_requests=len(test), linked_pairs=links,
                                        components=len(components), calibration_components=len(components[::3])),
                                    report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "glossary_file", "embedding", "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
