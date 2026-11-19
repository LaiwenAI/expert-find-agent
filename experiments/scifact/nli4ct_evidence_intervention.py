"""Separate shortlist misses from reader errors on annotated NLI4CT evidence."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.budgeted_sentence_selection import components
from experiments.scifact.nli4ct_adaptive_reader import choice, corpus, measure
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE
from experiments.scifact.nli4ct_verifier_traces import load_controller


ARMS = ("fixed_two", "gold_only", "gold_swap", "non_gold_swap")


def equal_size_swaps(sources, selected, gold):
    """Preserve original gold and replace the same non-gold indices in both views."""
    correct, control = [], []
    for (_, lines), kept, target in zip(sources, selected, gold, strict=True):
        missing = sorted(target - kept)
        if not missing:
            correct.append(kept)
            control.append(kept)
            continue
        retained_non_gold = kept - target
        alternatives = set(range(len(lines))) - kept - target
        if min(len(retained_non_gold), len(alternatives)) < len(missing):
            return None
        removed, added = set(), set()
        for idx in missing:
            # Match line length where possible without changing the number of lines.
            length = len(lines[idx])
            remove = min(retained_non_gold - removed, key=lambda j: (abs(len(lines[j]) - length), abs(j - idx), j))
            add = min(alternatives - added, key=lambda j: (abs(len(lines[j]) - length), abs(j - idx), j))
            removed.add(remove)
            added.add(add)
        original = kept - removed
        correct.append(frozenset(original | set(missing)))
        control.append(frozenset(original | added))
    return tuple(correct), tuple(control)


def views(sources, selected, gold):
    if any(not target <= set(range(len(lines))) for (_, lines), target in zip(sources, gold, strict=True)):
        raise ValueError("Gold line outside original sources")
    result = {"fixed_two": selected, "gold_only": gold}
    if any(not target <= kept for kept, target in zip(selected, gold, strict=True)):
        swapped = equal_size_swaps(sources, selected, gold)
        if swapped is not None:
            result.update(zip(ARMS[2:], swapped, strict=True))
    return result


def summarize(rows):
    report = {}
    for name, subset in (
        ("all", rows),
        ("fixed_complete", [r for r in rows if r["complete"]]),
        ("fixed_missing", [r for r in rows if not r["complete"]]),
        ("swappable_missing", [r for r in rows if "gold_swap" in r["views"]]),
    ):
        report[name] = {}
        for arm in ARMS:
            evaluated = [r for r in subset if arm in r["views"]]
            if not evaluated:
                continue
            binary = [r["views"][arm]["binary"] == r["label"] for r in evaluated]
            base = [r["views"]["fixed_two"]["binary"] == r["label"] for r in evaluated]
            delta = [int(a) - int(b) for a, b in zip(binary, base, strict=True)]
            report[name][arm] = dict(
                cases=len(evaluated), correct=sum(binary), changed_for_better=delta.count(1),
                changed_for_worse=delta.count(-1), abstentions=sum(r["views"][arm]["tri"] == 2 for r in evaluated),
                correct_with_abstention=sum(r["views"][arm]["tri"] == r["label"] for r in evaluated),
                lines=mean(r["views"][arm]["lines"] for r in evaluated),
                input_tokens=mean(r["views"][arm]["tokens"] for r in evaluated),
                reader_p95_ms=float(np.quantile([r["views"][arm]["ms"] for r in evaluated], .95)),
                delta_vs_fixed=clustered_paired_interval(delta, [r["group"] for r in evaluated])
                | {"sampling_unit": "connected trial component; query-weighted mean"},
            )
    subset = [r for r in rows if "gold_swap" in r["views"]]
    paired = [int(r["views"]["gold_swap"]["binary"] == r["label"])
              - int(r["views"]["non_gold_swap"]["binary"] == r["label"]) for r in subset]
    report["gold_vs_non_gold_swap"] = dict(
        cases=len(subset), gold_only_correct=paired.count(1), control_only_correct=paired.count(-1),
        delta=clustered_paired_interval(paired, [r["group"] for r in subset])
        | {"sampling_unit": "connected trial component; query-weighted mean"},
    )
    return report


def run(archive, frozen, checkpoint, output):
    import torch
    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm CUDA allocation")
    if digest(archive) != ARCHIVE:
        raise ValueError("Changed NLI4CT archive")
    controller, cutoff = load_controller(frozen)
    tokenizer, reader, provenance = load_reader(checkpoint)
    encoded = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 3:
        raise ValueError("Expected three distinct single-token labels")
    digits = [ids[0] for ids in encoded]
    cases = list(corpus(archive))
    links = [dict(case=row[0], doc=paper) for row in cases for paper, _ in row[4]]
    groups = {link["case"]: group for link, group in zip(links, components(links), strict=True)}

    def observe(query, section, sources, indices):
        start = perf_counter()
        binary, tri, tokens = measure(tokenizer, reader, digits, query, section, sources, indices)
        return dict(binary=binary, tri=tri, tokens=tokens, lines=sum(map(len, indices)),
                    ms=(perf_counter() - start) * 1000)

    key, query, section, _, sources, _ = cases[0]
    observe(query, section, sources, choice(key, query, sources, controller, cutoff, "fixed_two"))
    rows = []
    for i, (key, query, section, label, sources, gold) in enumerate(cases):
        selected = choice(key, query, sources, controller, cutoff, "fixed_two")
        choices = views(sources, selected, gold)
        complete = all(target <= kept for kept, target in zip(selected, gold, strict=True))
        if "gold_swap" in choices:
            assert [len(x) for x in selected] == [len(x) for x in choices["gold_swap"]] == [len(x) for x in choices["non_gold_swap"]]
            assert all(target <= kept for kept, target in zip(choices["gold_swap"], gold, strict=True))
            assert [target & kept for kept, target in zip(choices["non_gold_swap"], gold, strict=True)] == [
                target & kept for kept, target in zip(selected, gold, strict=True)
            ]
        rotation = i % len(ARMS)
        readings = {arm: observe(query, section, sources, choices[arm])
                    for arm in ARMS[rotation:] + ARMS[:rotation] if arm in choices}
        rows.append(dict(complete=complete, label={"Entailment":0, "Contradiction":1}[label],
                         group=groups[key], views=readings))
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1} requests", flush=True)
    if len(rows) != 500 or sum(row["complete"] for row in rows) != 264:
        raise ValueError("Changed fixed-depth cohort or evidence coverage")
    counts = Counter(len(row[4]) for row in cases)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            reader_revision=provenance["revision"], cases=len(rows), sources_per_case=counts,
                            archive_sha256=ARCHIVE, frozen_controller_sha256=digest(frozen),
                            script_sha256=digest(Path(__file__)), report=summarize(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "frozen", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
