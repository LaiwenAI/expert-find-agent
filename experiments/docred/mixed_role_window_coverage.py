"""Transfer learned proof windows to unseen titles and mixed-role requests."""

import argparse
from collections import defaultdict
from itertools import combinations, product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import learned_conjunctive_windows as learner
from experiments.followir.core17_reranking import digest, write_json


BASE = learner.base
NAMES = learner.names
ARMS = learner.ARMS
COVERAGES = (.35, .50, .65, 1.)
EXTERNAL_CAP = 100


def source_records(doc):
    entities = doc["vertexSet"]
    canonical = [NAMES.canonical(entity) for entity in entities]
    roles = BASE.shapes.roles(doc)
    evidence = {(label["h"], label["t"], label["r"]): tuple(sorted(set(label.get("evidence", ()))))
                for label in doc["labels"] if (label["h"], label["t"], label["r"]) in roles}
    return entities, canonical, evidence


def make_case(doc, canonical, evidence, requirements, people):
    unique_names = {tuple(NAMES.tokens(canonical[i]))
                    for i in (*[work for work, _ in requirements], *people)}
    if len(unique_names) != 4:
        return None
    cells = [(work, person, role) for (work, role), person in product(requirements, people)]
    labeled = [cell in evidence for cell in cells]
    annotated = [evidence.get(cell, ()) for cell in cells]
    return dict(
        doc=doc["title"], id=NAMES.stable(doc["title"], cells),
        names=[(canonical[work], canonical[person]) for work, person, _ in cells],
        roles=tuple(role for _, _, role in cells),
        sentences=[" ".join(row) for row in doc["sents"]],
        known=[all(labeled[i] for i in pair) for pair in BASE.PAIR],
        annotated=annotated,
    )


def mixed_cases(docs):
    for doc in docs:
        if len(doc["sents"]) < 4:
            continue
        _, canonical, evidence = source_records(doc)
        people_by_requirement = defaultdict(set)
        for work, person, role in evidence:
            people_by_requirement[work, role].add(person)
        chosen = {}
        for left, right in combinations(sorted(people_by_requirement), 2):
            if left[0] == right[0] or left[1] == right[1]:
                continue
            first, second = people_by_requirement[left], people_by_requirement[right]
            for a, b in combinations(sorted(first | second), 2):
                if not ((a in first and b in second) or (b in first and a in second)):
                    continue
                case = make_case(doc, canonical, evidence, (left, right), (a, b))
                if case is None or not any(case["annotated"]):
                    continue
                kind = any(case["known"])
                if kind not in chosen or case["id"] < chosen[kind]["id"]:
                    chosen[kind] = case
        yield from chosen.values()


def broader_training_cases(docs):
    for doc in docs:
        if len(doc["sents"]) < 4:
            continue
        entities, canonical, evidence = source_records(doc)
        people = [i for i, entity in enumerate(entities) if entity[0]["type"] == "PER"]
        works = [i for i, entity in enumerate(entities) if entity[0]["type"] != "PER"]
        credited = {work for work, _, _ in evidence}
        for (work, person, role), gold in sorted(evidence.items()):
            if not 0 < len(gold) <= 2:
                continue
            candidates = ((other_work, other_person) for other_work, other_person in product(works, people)
                          if other_work != work and other_person != person)
            ordered = sorted(candidates, key=lambda pair: (
                pair[0] not in credited, NAMES.stable(doc["title"], (work, person, role), pair)
            ))
            for other_work, other_person in ordered:
                case = make_case(doc, canonical, evidence, ((work, role), (other_work, role)),
                                 (person, other_person))
                if case is not None:
                    # This training example supervises only the actual anchor relation.
                    yield dict(case, annotated=[gold, (), (), ()])
                    break


def load(release):
    original, hashes = BASE.load(release)
    docs = {split: json.loads((release / f"{split}_revised.json").read_bytes())
            for split in ("train", "dev", "test")}
    official = {doc["title"] for split in ("dev", "test") for doc in docs[split]}
    old_holdout = {case["doc"] for case in original["external"]}
    pool = defaultdict(list)
    for case in mixed_cases(doc for doc in docs["train"] if doc["title"] not in official | old_holdout):
        pool[any(case["known"])].append(case)
    external = [case for kind in (False, True)
                for case in sorted(pool[kind], key=lambda row: row["id"])[:EXTERNAL_CAP]]
    mixed = dict(calibration=list(mixed_cases(docs["dev"])), test=list(mixed_cases(docs["test"])),
                 external=external)
    if not all(mixed.values()):
        raise ValueError("An independent mixed-role panel is empty")
    weights, training = learner.fit(release, {**original, "mixed_external": external},
                                    case_builder=broader_training_cases)
    return original, mixed, hashes, weights, training


def decisions(rows):
    result = {}
    for arm in ARMS:
        entries = []
        for case, records in rows:
            record = records[arm]
            margins = record["margins"]
            # Stable, identity-independent choice when candidate scores tie.
            first = int(NAMES.stable(case["id"], "candidate")[-1], 16) % 2
            top = first if margins[0] == margins[1] else int(margins[1] > margins[0])
            key = (float(margins[top]), NAMES.stable(case["id"], "coverage"))
            proof = all(record["available"][i] for i in BASE.PAIR[top])
            elapsed = record["reader_ms"] + record["selection_ms"]
            entries.append((key, bool(case["known"][top]), proof, elapsed))
        result[arm] = entries
    return result


def coverage_keys(entries, fraction):
    count = max(1, int(round(len(entries) * fraction)))
    return sorted((row[0] for row in entries), reverse=True)[count-1]


def summarize(rows, calibration):
    entries = decisions(rows)
    clusters = [case["doc"] for case, _ in rows]
    result = {}
    for kind in ("locked_development", "matched_test_label_free"):
        curves = {}
        for fraction in COVERAGES:
            arms = {}
            accepted = {}
            for arm, values in entries.items():
                cut = coverage_keys(calibration[arm] if kind == "locked_development" else values, fraction)
                chosen = np.array([fraction == 1. or key >= cut for key, *_ in values])
                hits = chosen & np.array([row[1] for row in values])
                citation = chosen & np.array([row[2] for row in values])
                arms[arm] = dict(issued=int(chosen.sum()), known=int(hits.sum()),
                                 unjudged=int(chosen.sum()-hits.sum()), with_proof=int(citation.sum()))
                accepted[arm] = hits.astype(int)
            for arm in arms:
                arms[arm]["paired_known_delta_vs_global"] = clustered_paired_interval(
                    accepted[arm] - accepted["global_names"], clusters,
                )
            curves[str(fraction)] = arms
        result[kind] = curves
    return dict(cases=len(rows), titles=len(set(clusters)),
                possible_known=sum(any(case["known"]) for case, _ in rows),
                combined_p95_ms={arm: float(np.quantile([row[3] for row in values], .95))
                                 for arm, values in entries.items()}, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm allocation")
    start = perf_counter()
    original, mixed, hashes, weights, training = load(args.release)
    print(f"Fitted {training}; mixed panels { {key: len(rows) for key, rows in mixed.items()} }", flush=True)
    training["fit_and_data_seconds"] = perf_counter() - start
    tokenizer, model, revision = BASE.reader.load_reader(args.checkpoint)
    read = BASE.reader.reader_scorer(tokenizer, model, system=BASE.shapes.SYSTEM)
    learner.collect(mixed["calibration"][:1], weights, read, 0)
    results = {}
    for family, panels in (("same_role", original), ("mixed_role", mixed)):
        calibration = decisions(learner.collect(panels["calibration"], weights, read, args.order_offset))
        for label in ("test", "external"):
            runs = learner.collect(panels[label], weights, read, args.order_offset)
            results[f"{family}_{label}"] = summarize(runs, calibration)
            print(f"Finished {family} {label}: {len(runs)} requests", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], inputs=hashes, training=training,
        weights=weights.tolist(), checkpoint=revision["revision"], order_offset=args.order_offset,
        script_sha256=digest(Path(__file__)), selector_sha256=digest(Path(learner.__file__)),
        groups=results,
    ))


if __name__ == "__main__":
    main()
