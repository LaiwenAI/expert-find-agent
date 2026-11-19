"""Calibrate low-cost support gates on separate VitaminC article pairs."""

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.nli4ct_adaptive_reader import label_scores
from experiments.scifact.vitaminc_label_transfer import DATA_SHA256, LABELS, REVISION, mapped_prompts, panel, stable


DEV_SHA256 = "544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5"
PROMPTS = ("original", "swap_both")
CUTS = (0., 1., 2., 4., 6., 8., 10., 14., 18.)
MAX_REVIEW_FRACTION = .25
PAIR_DEPTH = 200


@dataclass(frozen=True)
class Gate:
    start: int
    low: float
    high: float
    confirm: float = 0.

    def apply(self, margins):
        primary = margins[:, self.start]
        secondary = margins[:, 1 - self.start]
        review = (primary >= self.low) & (primary < self.high)
        vote = (primary >= self.high) | (review & (secondary >= self.confirm))
        return vote, review


def calibrate(margins, targets, mode):
    direct = [Gate(start, cut, cut) for start in range(2) for cut in CUTS]
    # Include a defer-all setting if none of the operating points can satisfy the cap.
    direct.extend(Gate(start, 1e6, 1e6) for start in range(2))
    cascade = [Gate(start, low, high, confirm)
               for start in range(2) for low in CUTS for high in CUTS if high > low for confirm in CUTS]
    gates = {
        "original": [g for g in direct if g.start == 0],
        "swap_both": [g for g in direct if g.start == 1],
        "selective": [*direct, *cascade],
    }[mode]
    baseline, _ = Gate(0, 0., 0.).apply(margins)
    caps = [int(np.sum(baseline & (targets == label))) for label in (1, 2)]
    feasible = []
    for gate in gates:
        accepted, reviewed = gate.apply(margins)
        errors = [int(np.sum(accepted & (targets == label))) for label in (1, 2)]
        n_reviewed = int(reviewed.sum())
        if all(e <= bound for e, bound in zip(errors, caps, strict=True)) and n_reviewed <= MAX_REVIEW_FRACTION * len(targets):
            positives = int(np.sum(accepted & (targets == 0)))
            feasible.append((-positives, n_reviewed, sum(errors), gate.start, gate.low, gate.high, gate.confirm, gate))
    return min(feasible)[-1], caps


def statistics(gate, margins, targets, durations, groups):
    support, review = gate.apply(margins)
    base, _ = Gate(0, 0., 0.).apply(margins)
    results = {}
    for label, name in enumerate(("true_support", "false_support_on_refute", "false_support_on_insufficient")):
        eligible = targets == label
        selected_groups = [group for group, match in zip(groups, eligible, strict=True) if match]
        diff = support[eligible].astype(int) - base[eligible].astype(int)
        results[name] = {
            "accepted": int(support[eligible].sum()), "total": int(eligible.sum()),
            "delta_vs_original": clustered_paired_interval(diff, selected_groups)
            | {"sampling_unit": "one contrast pair per Wikipedia page"},
        }
    both = ((targets == 0) == support).reshape(-1, 2).all(axis=1)
    both_base = ((targets == 0) == base).reshape(-1, 2).all(axis=1)
    duration = durations[:, gate.start] + durations[:, 1 - gate.start] * review
    return dict(results, reviewed=int(review.sum()), total=len(margins), average_forwards=1 + float(review.mean()),
                pair_support_decisions_correct=int(both.sum()),
                paired_both_correct_delta=clustered_paired_interval(both.astype(int)-both_base.astype(int), groups[::2])
                | {"sampling_unit": "one contrast pair per Wikipedia page"},
                factored_p50_p95_ms=np.quantile(duration, [.5, .95]).tolist())


def cohorts(dev, test):
    explored = panel(test)
    old_pages = {a["page"] for a, _ in explored}
    old_claims = {a["claim"].casefold() for a, _ in explored}
    held = panel(test, seed="vitaminc-selective-v1-test", per_control=PAIR_DEPTH,
                 excluded_pages=old_pages, excluded_claims=old_claims)
    current_pages = {a["page"] for a, _ in held}
    current_claims = {a["claim"].casefold() for a, _ in held}
    development = panel(dev, seed="vitaminc-selective-v1-dev", per_control=PAIR_DEPTH,
                        excluded_pages=old_pages | current_pages, excluded_claims=old_claims | current_claims)
    return development, held, old_pages, old_claims


def run(dev, test, checkpoint, output):
    import torch
    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA allocation")
    if digest(dev) != DEV_SHA256 or digest(test) != DATA_SHA256:
        raise ValueError("Changed pinned VitaminC file")
    development, held, old_pages, _ = cohorts(dev, test)
    tokenizer, reader, provenance = load_reader(checkpoint)
    digits = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
    if any(len(ids) != 1 for ids in digits) or len({ids[0] for ids in digits}) != 3:
        raise ValueError("Expected three distinct single-token labels")
    prompts = mapped_prompts(digits)

    def measure(row, arm):
        prompt, ids = prompts[arm]
        start = perf_counter()
        scores, _ = label_scores(tokenizer, reader, ids, row["claim"], "Wikipedia excerpt",
                                 (("", [row["evidence"]]),), (frozenset({0}),), prompt=prompt,
                                 headings=("Evidence",))
        return float(scores[0] - max(scores[1:])), (perf_counter() - start) * 1000

    for arm in PROMPTS:
        measure(development[0][0], arm)

    def collect(pairs, split):
        margins, durations, targets, groups = [], [], [], []
        for i, pair in enumerate(pairs):
            for j, row in enumerate(pair):
                order = PROMPTS if (i + j) % 2 == 0 else PROMPTS[::-1]
                readings = {arm: measure(row, arm) for arm in order}
                margins.append([readings[arm][0] for arm in PROMPTS])
                durations.append([readings[arm][1] for arm in PROMPTS])
                targets.append(LABELS.index(row["label"]))
                groups.append(row["page"])
            if (i + 1) % 100 == 0:
                print(split, f"completed {i+1} of {len(pairs)} pairs", flush=True)
        return np.asarray(margins), np.asarray(targets), np.asarray(durations), groups

    dev_rows = collect(development, "development")
    gates = {"original_raw": Gate(0, 0., 0.), "swap_raw": Gate(1, 0., 0.)}
    for mode in ("original", "swap_both", "selective"):
        gate, caps = calibrate(*dev_rows[:2], mode)
        gates[mode + "_calibrated"] = gate
    test_rows = collect(held, "test")
    report = {name: dict(gate=asdict(gate), development=statistics(gate, *dev_rows),
                         test=statistics(gate, *test_rows)) for name, gate in gates.items()}
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reader_revision=provenance["revision"],
        dataset_revision=REVISION, dev_sha256=DEV_SHA256, test_sha256=DATA_SHA256,
        script_sha256=digest(Path(__file__)), control_file_sha256=digest(Path(mapped_prompts.__code__.co_filename)),
        dev_panel_sha256=stable(*(r["unique_id"] for pair in development for r in pair)).hex(),
        test_panel_sha256=stable(*(r["unique_id"] for pair in held for r in pair)).hex(),
        excluded_explored_pages=len(old_pages), max_dev_review_fraction=MAX_REVIEW_FRACTION,
        dev_false_support_caps=caps, pairs_per_negative_class=PAIR_DEPTH, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("dev", "test", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
