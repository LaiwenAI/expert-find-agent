"""Test same-role ownership across multiple works with updated Re-DocRED labels."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.docred import oracle_role_evidence as reader
from experiments.docred import person_bound_evidence as base
from experiments.followir.core17_reranking import digest, write_json


UPDATED = {
    "dev": "ccc81b3c34b741c6ff12993edd6672b549b4e960cfb7180f475ee3d60ff659cc",
    "test": "7630b5afc39b3d9a9bece9e0d910d157fb4430689d765ce71ff76439ab27c3f0",
}
ARMS = ("full", "shared_four")
GATES = ("absolute", "other_work", "other_person", "both")
BUDGETS = (.05, .10)
SYSTEM = base.TASK + " The same person can have the same role on multiple works, and multiple people can hold the same role. Reply 1 for explicit support or 0 otherwise."


def documents(original, extended, split):
    old = original / f"{split}_revised.json"
    new = extended / f"re2docred_{split}_re.json"
    if digest(old) != base.DATA[split] or digest(new) != UPDATED[split]:
        raise ValueError(f"Changed annotation snapshot for {split}")
    updates = {d["title"]: d for d in json.loads(new.read_bytes())}
    for doc in json.loads(old.read_bytes()):
        revised = updates[doc["title"]]
        if doc["sents"] != revised["sents"] or doc["vertexSet"] != revised["vertexSet"]:
            raise ValueError("Released entity indices or source text differ")
        yield doc, revised


def roles(document):
    entities = document["vertexSet"]
    return {(l["h"], l["t"], l["r"]) for l in document["labels"] if l["r"] in base.ROLES
            and entities[l["t"]][0]["type"] == "PER" and entities[l["h"]][0]["type"] != "PER"}


def cases(original, extended, split):
    yield from cases_from_documents(documents(original, extended, split))


def cases_from_documents(document_pairs):
    for doc, newer in document_pairs:
        old = roles(doc)
        updated = roles(newer)
        if not old <= updated:
            raise ValueError("A published positive was removed")
        groups = defaultdict(lambda: defaultdict(set))
        for work, person, relation in sorted(old):
            groups[relation][work].add(person)
        names = [base.canonical(entity) for entity in doc["vertexSet"]]
        candidates = defaultdict(list)
        for relation, works in sorted(groups.items()):
            for left, right in combinations(sorted(works), 2):
                for first, second in combinations(sorted(works[left] | works[right]), 2):
                    cells = tuple((work, person, relation) for work, person in product((left, right), (first, second)))
                    flags = tuple(cell in old for cell in cells)
                    if (not (flags[0] and flags[3] or flags[1] and flags[2])
                            or len({tuple(base.tokens(names[i])) for i in (left, right, first, second)}) != 4):
                        continue
                    key = base.stable(doc["title"], cells)
                    candidates[sum(flags)].append((key, cells, flags))
        for count in sorted(candidates):
            key, cells, flags = min(candidates[count])
            yield dict(
                doc=doc["title"], id=key, role=cells[0][2],
                names=[(names[work], names[person]) for work, person, _ in cells],
                sentences=[" ".join(sentence) for sentence in doc["sents"]],
                prior=flags, labeled=tuple(cell in updated for cell in cells),
            )


def window(case):
    words, _, idf = base.profile(case["sentences"])
    names = list(dict.fromkeys(name for pair in case["names"] for name in pair))
    terms = [tuple(base.tokens(name)) for name in names]
    values = [[max(float(base.contains(row, name)), .6 * base.coverage(name, row, idf)) for name in terms]
              + [base.coverage(base.tokens(base.ROLE_CUES[case["role"]]), row, idf)] for row in words]
    available = set(range(len(words)))
    chosen = []
    covered = [0.] * len(values[0])
    for _ in range(min(4, len(words))):
        best = max(available, key=lambda i: (sum(max(v - c, 0.) for v, c in zip(values[i], covered, strict=True)),
                                            sum(values[i]), -i))
        chosen.append(best)
        covered = list(map(max, zip(covered, values[best], strict=True)))
        available.remove(best)
    return "\n".join(case["sentences"][i] for i in sorted(chosen))


def observe(case, read, arm):
    start = perf_counter()
    context = "\n".join(case["sentences"]) if arm == "full" else window(case)
    prompts = [(f"Does the source explicitly say that {person} had the role of {base.ROLES[case['role']]} "
                f"on the work {work}?", context) for work, person in case["names"]]
    scores, clipped = read(prompts)
    ms = (perf_counter() - start) * 1000
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid reader score")
    return dict(scores=scores, ms=ms, words=len(context.split()))


def rank_scores(scores, rule):
    if rule == "absolute":
        return scores
    other_work = [score - scores[i ^ 2] for i, score in enumerate(scores)]
    other_person = [score - scores[i ^ 1] for i, score in enumerate(scores)]
    if rule == "other_work":
        return other_work
    if rule == "other_person":
        return other_person
    return list(map(min, zip(other_work, other_person, strict=True)))


def calibrate(rows, arm, rule, budget):
    unknown = [s for case, measured in rows for s, positive in
               zip(rank_scores(measured[arm]["scores"], rule), case["labeled"], strict=True) if not positive]
    unknown.sort(reverse=True)
    return unknown[int(len(unknown) * budget)]


def summarize(rows, arm, rule, cutoff):
    counts = Counter()
    for case, measured in rows:
        labeled = case["labeled"]
        scores = rank_scores(measured[arm]["scores"], rule)
        accepted = [s > cutoff for s in scores]
        counts["rectangles"] += 1
        counts["known"] += sum(labeled)
        counts["unjudged"] += 4 - sum(labeled)
        counts["retained_known"] += sum(a and k for a, k in zip(accepted, labeled, strict=True))
        counts["issued_unjudged"] += sum(a and not k for a, k in zip(accepted, labeled, strict=True))
        counts["all_known_retained"] += all(a or not k for a, k in zip(accepted, labeled, strict=True))
        counts["exact_label_pattern"] += tuple(accepted) == labeled
        counts["newly_annotated"] += sum(b and not a for a, b in zip(case["prior"], labeled, strict=True))
        counts["newly_annotated_retained"] += sum(a and k and not o for a, k, o in
                                                  zip(accepted, labeled, case["prior"], strict=True))
    return dict(counts)


def run(original, extended, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    panels = {split: list(cases(original, extended, split)) for split in ("dev", "test")}
    if {c["doc"] for c in panels["dev"]} & {c["doc"] for c in panels["test"]}:
        raise ValueError("Calibration and test documents intersect")
    tokenizer, model, revision = reader.load_reader(checkpoint)
    score = reader.reader_scorer(tokenizer, model, system=SYSTEM)
    for arm in ARMS:
        observe(panels["dev"][0], score, arm)
    rows = {}
    for split, panel in panels.items():
        rows[split] = [(case, {arm: observe(case, score, arm) for arm in (ARMS if i % 2 == 0 else ARMS[::-1])})
                       for i, case in enumerate(panel)]
    groups = dict(combined=rows["test"], disjoint=[r for r in rows["test"] if sum(r[0]["labeled"]) == 2],
                  shared=[r for r in rows["test"] if sum(r[0]["labeled"]) > 2])
    report = {}
    for budget in BUDGETS:
        policies = report[str(budget)] = {}
        for arm in ARMS:
            policies[arm] = {}
            for rule in GATES:
                cutoff = calibrate(rows["dev"], arm, rule, budget)
                policies[arm][rule] = dict(cutoff=cutoff, calibration=summarize(rows["dev"], arm, rule, cutoff),
                                           test={group: summarize(items, arm, rule, cutoff)
                                                 for group, items in groups.items()})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), checkpoint=revision["revision"],
        input_sha256=dict(original=base.DATA, extended=UPDATED), reports=report,
        rectangles={k: len(v) for k, v in rows.items()},
        distinct_documents={k: len({c["doc"] for c, _ in v}) for k, v in rows.items()},
        test_profile={arm: dict(p95_ms=float(np.quantile([m[arm]["ms"] for _, m in rows["test"]], .95)),
                               mean_words=float(np.mean([m[arm]["words"] for _, m in rows["test"]])),
                               model_pairs=4 * len(rows["test"])) for arm in ARMS}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "extended", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
