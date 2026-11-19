"""Evaluate the SciFact-Open selectors on NLI4CT cross-trial evidence sets."""

import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

from expert_search.evidence_windows import choose_windows
from expert_search.statistics import clustered_paired_interval
from experiments.scifact.budgeted_sentence_selection import ARMS, candidate_indices, components, digest


ARCHIVE = "dd972c46e69a5bc5e5fb74c2105b5444722d81879ab96b214545b943e4e7ebbe"


def collect(archive):
    with ZipFile(archive) as source:
        cases = json.loads(source.read("Complete_dataset/Gold_test.json"))
        trials = {}
        audits = []
        for key, row in sorted(cases.items()):
            sources = ("Primary",) if row["Type"] == "Single" else ("Primary", "Secondary")
            fragments = []
            for side in sources:
                paper = row[f"{side}_id"]
                if paper not in trials:
                    trials[paper] = json.loads(source.read(f"Complete_dataset/CT json/{paper}.json"))
                sentences = trials[paper][row["Section_id"]]
                gold = set(row[f"{side}_evidence_index"])
                if not gold or not 0 <= min(gold) <= max(gold) < len(sentences):
                    raise ValueError("Invalid gold evidence index")
                claim = row["Statement"]
                _, base = choose_windows(claim, sentences)
                chosen = {"lexical_windows": set(base)} | {
                    arm: set(candidate_indices(claim, sentences, len(base), arm)) for arm in ARMS[1:]
                }
                fragments.append(dict(trial=paper, budget=len(base), gold=gold, chosen=chosen))
            audits.append(dict(case=key, kind=row["Type"], section=row["Section_id"], fragments=fragments))
    return audits


def summarize(cases, groups, arm):
    baseline = "lexical_windows"
    full, hits = [], 0
    totals = Counter()
    for row in cases:
        complete = {method: all(fr["gold"] <= fr["chosen"][method] for fr in row["fragments"])
                    for method in (baseline, arm)}
        hits += complete[arm]
        full.append(int(complete[arm]) - int(complete[baseline]))
        for fr in row["fragments"]:
            totals["gold"] += len(fr["gold"])
            totals["chosen"] += len(fr["chosen"][arm])
            totals["matched"] += len(fr["gold"] & fr["chosen"][arm])
        totals["feasible"] += all(len(fr["gold"]) <= fr["budget"] for fr in row["fragments"])
        totals["feasible_shared_budget"] += (
            sum(len(fr["gold"]) for fr in row["fragments"]) <= sum(fr["budget"] for fr in row["fragments"])
        )
    return dict(
        cases=len(cases), complete_all_trials=hits, feasible_all_trials=totals["feasible"],
        feasible_shared_budget=totals["feasible_shared_budget"],
        paired_difference=clustered_paired_interval(full, [groups[row["case"]] for row in cases]),
        sentence_micro_f1=2 * totals["matched"] / (totals["gold"] + totals["chosen"]),
        mean_selected_sentences=totals["chosen"] / len(cases),
    )


def run(archive, output):
    if digest(archive) != ARCHIVE or output.exists():
        raise ValueError("Require the pinned archive and a new result path")
    audits = collect(archive)
    if len(audits) != 500:
        raise ValueError("Unexpected official test split size")
    edges = [dict(case=row["case"], doc=fr["trial"]) for row in audits for fr in row["fragments"]]
    groups = {edge["case"]: group for edge, group in zip(edges, components(edges), strict=True)}
    slices = {
        "all": audits,
        "single_trial": [row for row in audits if row["kind"] == "Single"],
        "two_trials": [row for row in audits if row["kind"] == "Comparison"],
    }
    report = dict(
        archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
        selector_sha256=digest(Path(__file__).with_name("budgeted_sentence_selection.py")),
        results={name: {arm: summarize(rows, groups, arm) for arm in ARMS}
                 for name, rows in slices.items()},
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
