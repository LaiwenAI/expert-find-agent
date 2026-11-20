"""Allocate a fixed evidence window to two same-person work requirements."""

import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import oracle_role_evidence as reader
from experiments.docred import person_bound_evidence as names
from experiments.docred import work_owner_contrast as shapes
from experiments.docred.scope_triggered_recheck import TRAIN_SHA256
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("global_names", "claim_quota", "claim_coverage", "conjunction")
WIDTH = 4
PAIR = (0, 2), (1, 3)
DEV_RISK = .05
TEST_CAP = 120


def cases(documents):
    for document in documents:
        evidence = defaultdict(list)
        canonical = [names.canonical(entity) for entity in document["vertexSet"]]
        for item in document["labels"]:
            evidence[canonical[item["h"]], canonical[item["t"]], item["r"]].append(
                tuple(sorted(set(item.get("evidence", ()))))
            )
        for case in shapes.cases_from_documents([(document, document)]):
            if len(case["sentences"]) < WIDTH:
                continue
            claims = [evidence.get((work, person, case["role"]), []) for work, person in case["names"]]
            if any(len(matched) != int(labeled)
                   for matched, labeled in zip(claims, case["labeled"], strict=True)):
                continue
            annotated = tuple(matched[0] if matched else () for matched in claims)
            known = tuple(all(case["labeled"][i] for i in indexes) for indexes in PAIR)
            # Require some evidence to be available for the sentence-budget diagnostic.
            if not any(annotated):
                continue
            yield dict(case, annotated=annotated, known=known)


def load(root):
    hashes = {**names.DATA, "train": TRAIN_SHA256["original"]}
    docs = {}
    for split, expected in hashes.items():
        path = root / f"{split}_revised.json"
        if digest(path) != expected:
            raise ValueError(f"Changed Re-DocRED release: {split}")
        docs[split] = json.loads(path.read_bytes())
    held = {doc["title"] for split in ("dev", "test") for doc in docs[split]}
    pools = {"calibration": list(cases(docs["dev"])), "test": list(cases(docs["test"]))}
    groups = defaultdict(list)
    for case in cases(doc for doc in docs["train"] if doc["title"] not in held):
        groups[int(any(case["known"]))].append(case)
    pools["external"] = [
        case for kind in (0, 1)
        for case in sorted(groups[kind], key=lambda case: case["id"])[:TEST_CAP]
    ]
    for key, panel in pools.items():
        if not panel or len({case["id"] for case in panel}) != len(panel):
            raise ValueError(f"Invalid case pool: {key}")
    return pools, hashes


def features(case):
    words, _, idf = names.profile(case["sentences"])
    role = tuple(names.tokens(names.ROLE_CUES[case["role"]]))
    entities = {name: tuple(names.tokens(name)) for pair in case["names"] for name in pair}
    values = {
        name: np.array([max(float(names.contains(sentence, tokens)),
                            .6 * names.coverage(tokens, sentence, idf)) for sentence in words])
        for name, tokens in entities.items()
    }
    cue = np.array([names.coverage(role, sentence, idf) for sentence in words])
    global_names = np.column_stack([*values.values(), cue])
    scoped = np.column_stack([
        .65*np.minimum(values[work], values[person]) + .2*np.maximum(values[work], values[person]) + .15*cue
        for work, person in case["names"]
    ])
    return global_names, scoped


def weakest(covered):
    return np.minimum(covered[..., 0], covered[..., 2]) + np.minimum(covered[..., 1], covered[..., 3])


def choose_windows(group, scoped, arm):
    n = len(scoped)
    values = group if arm == "global_names" else scoped
    selected = []
    if arm == "claim_quota":
        selected.extend(dict.fromkeys(scoped.argmax(axis=0).tolist()))
    if arm == "conjunction":
        # A joint pair avoids the zero marginal gain of the first piece of a proof.
        left, right = np.triu_indices(n, 1)
        trial = np.maximum(scoped[left], scoped[right])
        best = np.lexsort((right, left, -trial.sum(axis=1), -weakest(trial)))[0]
        selected.extend((int(left[best]), int(right[best])))
    covered = values[selected].max(axis=0) if selected else np.zeros(values.shape[1])
    while len(selected) < WIDTH:
        available = [i for i in range(n) if i not in selected]
        possible = np.maximum(values[available], covered)
        if arm == "conjunction":
            best = np.lexsort((available, -possible.sum(axis=1), -weakest(possible)))[0]
        else:
            best = int(np.argmax(possible.sum(axis=1)))
        selected.append(available[best])
        covered = possible[best]
    return sorted(selected)


def assign_windows(scoped, selected):
    # Guarantee that the four chosen sentences are actually shown. Each
    # claim keeps its strongest sentence and receives one additional sentence.
    order = sorted(selected)
    first = [max(order, key=lambda s: scoped[s, i]) for i in range(4)]
    missing = sorted(set(order) - set(first))
    extra = [max((s for s in order if s != top), key=lambda s: scoped[s, i])
             for i, top in enumerate(first)]
    if missing:
        slots = max(permutations(range(4), len(missing)), key=lambda assignment: sum(
            scoped[s, i] - scoped[extra[i], i] for s, i in zip(missing, assignment, strict=True)
        ))
        for sentence, slot in zip(missing, slots, strict=True):
            extra[slot] = sentence
    return [tuple(sorted(pair)) for pair in zip(first, extra, strict=True)]


def measure(case, read, arm, signals):
    start = perf_counter()
    group, scoped = signals
    selected = choose_windows(group, scoped, arm)
    per_claim = assign_windows(scoped, selected)
    if set().union(*map(set, per_claim)) != set(selected) or any(len(set(ids)) != 2 for ids in per_claim):
        raise AssertionError("Evidence budgets differ")
    snippets = ["\n".join(case["sentences"][i] for i in rows) for rows in per_claim]
    claim_scores, clipped = read([
        (f"Does the source explicitly say that {person} had the role of {names.ROLES[case['role']]} "
         f"on the work {work}?", text)
        for (work, person), text in zip(case["names"], snippets, strict=True)
    ])
    if clipped or not np.isfinite(claim_scores).all():
        raise ValueError("Invalid relation judgment")
    margins = [min(claim_scores[i] for i in pair) for pair in PAIR]
    ms = (perf_counter() - start) * 1000
    available = [bool(gold) and set(gold) <= set(actual)
                 for gold, actual in zip(case["annotated"], per_claim, strict=True)]
    complete = [all(available[i] for i in pair) for pair in PAIR]
    gold_candidates = [all(case["annotated"][i] for i in pair) for pair in PAIR]
    return dict(
        margins=margins, complete=complete, gold_candidates=gold_candidates,
        atom_available=sum(available), atom_has_evidence=sum(bool(gold) for gold in case["annotated"]),
        annotated_inside_global=sum(bool(gold) and set(gold) <= set(selected) for gold in case["annotated"]),
        ms=ms, words=sum(len(text.split()) for text in snippets),
    )


def collect(panel, read, shift):
    results = []
    for i, case in enumerate(panel):
        begin = perf_counter()
        signals = features(case)
        shared_ms = (perf_counter() - begin) * 1000
        offset = (shift + i) % len(ARMS)
        observations = {}
        for arm in ARMS[offset:]+ARMS[:offset]:
            row = measure(case, read, arm, signals)
            row["ms"] += shared_ms
            observations[arm] = row
        results.append((case, observations))
        if (i + 1) % 60 == 0:
            print(f"Completed {i+1}/{len(panel)}", flush=True)
    return results


def cutoff(rows, arm):
    unknown = sorted((margin for case, reads in rows
                      for margin, known in zip(reads[arm]["margins"], case["known"], strict=True)
                      if not known), reverse=True)
    if not unknown:
        raise ValueError("No unjudged calibration candidates")
    return unknown[int(len(unknown)*DEV_RISK)]


def summarize(rows, thresholds):
    metrics, deltas = {}, {}
    clusters = [case["doc"] for case, _ in rows]
    for arm, decision in thresholds.items():
        totals = Counter()
        retained = []
        times = []
        for case, records in rows:
            row = records[arm]
            accepted = [value > decision for value in row["margins"]]
            known = case["known"]
            positives = sum(a and k for a,k in zip(accepted, known, strict=True))
            retained.append(positives)
            totals.update(
                known_candidates=sum(known), accepted_known=positives,
                unjudged_candidates=2-sum(known), accepted_unjudged=sum(
                    a and not k for a,k in zip(accepted, known, strict=True)
                ),
                candidates_with_annotated_proof=sum(row["gold_candidates"]),
                complete_annotated_proof=sum(row["complete"]),
                accepted_known_and_complete_annotated_proof=sum(
                    a and k and c for a,k,c in zip(accepted,known,row["complete"],strict=True)
                ),
                **{key: row[key] for key in ("atom_available", "atom_has_evidence", "annotated_inside_global", "words")},
            )
            times.append(row["ms"])
        deltas[arm] = np.array(retained, dtype=float) / 2
        metrics[arm] = dict(**totals, p95_ms=float(np.quantile(times, .95)), cutoff=decision)
    for arm in ARMS:
        metrics[arm]["paired_known_delta_vs_claim_coverage"] = clustered_paired_interval(
            deltas[arm] - deltas["claim_coverage"], clusters,
        )
    return dict(cases=len(rows), documents=len(set(clusters)), reads=WIDTH * len(rows),
                sentence_placements=8 * len(rows), arms=metrics)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pools, hashes = load(args.release)
    tokenizer, model, revision = reader.load_reader(args.checkpoint)
    score = reader.reader_scorer(tokenizer, model, system=shapes.SYSTEM)
    collect(pools["calibration"][:1], score, 0)
    outcomes = {name: collect(panel, score, args.order_offset) for name, panel in pools.items()}
    thresholds = {arm: cutoff(outcomes["calibration"], arm) for arm in ARMS}
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=revision["revision"],
        input_hashes=hashes, script_sha256=digest(Path(__file__)), order_offset=args.order_offset,
        groups={name: summarize(rows, thresholds) for name, rows in outcomes.items()},
    ))


if __name__ == "__main__":
    main()
