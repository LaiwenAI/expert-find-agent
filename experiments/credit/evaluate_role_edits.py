"""Evaluate evidence scope, paired agreement and abstention on contributor edits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.counterfactual_role_edits import (
    NEGATIVE, POSITIVE, UNKNOWN, VERSION, fingerprint, label, render,
)
from experiments.credit.plos_role_inventory import ROLES
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader


METHODS = ("all_evidence_reader", "scoped_reader", "scoped_hybrid")
CONDITIONS = ("canonical", "descriptive", "redacted")
MEANINGS = {
    POSITIVE: "The complete declaration for the target author lists both requested activities.",
    NEGATIVE: "At least one requested activity is omitted from the target author's complete declaration.",
    UNKNOWN: "The target author's complete declaration is unavailable.",
}
CANONICAL = re.compile(r"Does the statement credit Author (\d+) with both (.+) and (.+)\?")


def structured(question, author, roles):
    if roles is None:
        return UNKNOWN
    parsed = CANONICAL.fullmatch(question)
    if parsed is None:
        return None
    ordinal, first, second = parsed.groups()
    if int(ordinal) != author or first == second or first not in ROLES or second not in ROLES:
        return None
    return label(roles, (first, second))


def prompt(question, author, visible, options):
    return (
        "Determine what the contribution statement declares about the named author. "
        "Use only their declaration when making your choice.\n"
        + "\n".join(f"{token}: {MEANINGS[meaning]}" for token, meaning in zip(LABELS, options, strict=True))
        + f"\nTarget: Author {author}\nQuestion: {question}\n"
        + "\n".join(f"Complete declaration for Author {person}: {', '.join(roles)}" for person, roles in visible)
        + "\nAnswer:"
    )


def examples(case, all_roles, condition):
    style = "descriptive" if condition == "redacted" else condition
    result = []
    for prefix in ("changed", "stable"):
        author = case[f"{prefix}_author"]
        declaration = None if condition == "redacted" else all_roles[author - 1]
        for view, expected in zip(("before", "after"), case[f"{prefix}_labels"], strict=True):
            required = case[view]
            if label(all_roles[author - 1], required) != expected:
                raise ValueError(f"Invalid published label in {case['id']}")
            result.append((render(author, required, style), author, declaration,
                           UNKNOWN if declaration is None else expected))
    return result


def classify(reader, rows, paper, method, options):
    start = perf_counter()
    answers, requests, pending = [], [], []
    for i, (question, author, target, _) in enumerate(rows):
        result = structured(question, author, target) if method == "scoped_hybrid" else None
        answers.append(result)
        if result is None:
            if method == "all_evidence_reader":
                visible = [(k, role_set) for k, role_set in enumerate(paper, 1) if k != author or target is not None]
            else:
                visible = [(author, target)] if target is not None else []
            pending.append(i)
            requests.append(prompt(question, author, visible, options))
    tokens = 0
    if requests:
        decisions, _ = reader.classify(requests)
        for i, (token, length) in zip(pending, decisions, strict=True):
            answers[i] = options[LABELS.index(token)]
            tokens += length
    return dict(answers=answers, ms=(perf_counter() - start) * 1000, tokens=tokens, calls=int(bool(requests)))


def summarize(results):
    summaries = defaultdict(dict)
    for condition in CONDITIONS:
        subset = [row for row in results if row["condition"] == condition]
        papers = [row["doi"] for row in subset]
        for method in METHODS:
            votes = [entry["arms"][method] for entry in subset]
            expected = [entry["gold"] for entry in subset]
            differences = [int(v["answers"] == g) - int(row["arms"]["all_evidence_reader"]["answers"] == g)
                           for row, g, v in zip(subset, expected, votes, strict=True)]
            summaries[condition][method] = dict(
                edit_groups=len(votes),
                correct_group=sum(v["answers"] == g for v, g in zip(votes, expected, strict=True)),
                correct_changed_pair=sum(v["answers"][:2] == g[:2] for v, g in zip(votes, expected, strict=True)),
                correct_stable_pair=sum(v["answers"][2:] == g[2:] for v, g in zip(votes, expected, strict=True)),
                false_promotions=sum(p == POSITIVE and y != POSITIVE
                                     for v, g in zip(votes, expected, strict=True)
                                     for p, y in zip(v["answers"], g, strict=True)),
                group_accuracy_delta=clustered_paired_interval(differences, papers),
                mean_input_tokens=mean(v["tokens"] for v in votes),
                reader_calls=sum(v["calls"] for v in votes),
                latency_p50_p95_ms=np.quantile([v["ms"] for v in votes], [.5, .95]).tolist(),
            )
    return summaries


def run(fixture, checkpoint, output, split):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    path = fixture / "pairs.jsonl"
    if manifest["version"] != VERSION or digest(path) != manifest["pairs_sha256"]:
        raise ValueError("Unexpected counterfactual fixture")
    papers = {p["doi"]: p["roles_by_author"] for p in manifest["sources"] if p["split"] == split}
    groups = [g for line in path.read_text().splitlines() if (g := json.loads(line))["split"] == split]
    if not groups or {g["doi"] for g in groups} != set(papers):
        raise ValueError("Selected split doesn't match source manifest")
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Duplicate edit group")
    tokenizer, model, provenance = load_reader(checkpoint)
    reader = Reader(tokenizer, model)
    results = []
    for i, case in enumerate(groups):
        order = sorted(MEANINGS, key=lambda meaning: fingerprint(case["id"], meaning))
        paper = papers[case["doi"]]
        style_shift = i % len(CONDITIONS)
        for condition in CONDITIONS[style_shift:] + CONDITIONS[:style_shift]:
            rows = examples(case, paper, condition)
            if i == 0:
                for method in METHODS:
                    classify(reader, rows, paper, method, order)
            arm_shift = i % len(METHODS)
            arms = {method: classify(reader, rows, paper, method, order)
                    for method in METHODS[arm_shift:] + METHODS[:arm_shift]}
            results.append(dict(id=case["id"], doi=case["doi"], condition=condition,
                                gold=[row[-1] for row in rows], arms=arms))
    payload = dict(split=split, source_count=len(papers), pairs_sha256=digest(path), script_sha256=digest(Path(__file__)),
                   checkpoint=provenance, summary=summarize(results), results=results)
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()
    run(args.fixture, args.checkpoint, args.output, args.split)
