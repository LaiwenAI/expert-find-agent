"""Measure and calibrate output-label bias with fixed evidence and one-call inference."""

import argparse
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.budgeted_sentence_selection import components
from experiments.scifact.nli4ct_adaptive_reader import PROMPT, choice, corpus, label_scores
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE
from experiments.scifact.nli4ct_verifier_traces import load_controller


VIEWS = ("fixed_two", "gold_only")
MAPPINGS = ("original", "swap_tokens", "swap_order", "swap_both")
OLD = "Output 0 for entailment, 1 for contradiction, or 2 if the supplied text is insufficient."


def prompt_variant(mapping, digits):
    a, b = (1, 0) if mapping in ("swap_tokens", "swap_both") else (0, 1)
    categories = [f"{a} for entailment", f"{b} for contradiction"]
    if mapping in ("swap_order", "swap_both"):
        categories.reverse()
    replacement = f"Output {categories[0]}, {categories[1]}, or 2 if the supplied text is insufficient."
    assert OLD in PROMPT
    return PROMPT.replace(OLD, replacement), [digits[a], digits[b], digits[2]]


def summarize(rows, correction, groups):
    outcomes = {}
    margins = {arm: np.asarray([r["readings"][arm]["margin"] for r in rows]) for arm in MAPPINGS}
    labels = np.asarray([r["target"] for r in rows])
    answers = margins["original"] >= 0.
    policies = {
        **margins,
        "single_call_correction": margins["original"] + correction,
        "two_mappings_average": (margins["original"] + margins["swap_tokens"]) / 2,
        "four_variants_average": np.mean(list(margins.values()), axis=0),
    }
    for arm, values in policies.items():
        predictions = values >= 0.
        correct = predictions == labels
        delta = correct.astype(int) - (answers == labels).astype(int)
        outcomes[arm] = dict(
            correct=int(sum(correct)), entailment_predictions=int(sum(predictions)),
            entailment_correct=int(sum(correct & labels)), contradiction_correct=int(sum(correct & ~labels)),
            changes_vs_original=int(sum(predictions != answers)),
            mean_margin=float(values.mean()),
            accuracy_delta_vs_original=clustered_paired_interval(delta, groups)
            | {"sampling_unit": "connected trial component; query-weighted mean"},
        )
        if arm in MAPPINGS:
            outcomes[arm]["mean_tokens"] = mean(r["readings"][arm]["tokens"] for r in rows)
            outcomes[arm]["p95_ms"] = float(np.quantile([r["readings"][arm]["ms"] for r in rows], .95))
    return outcomes


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
    prompts = {arm: prompt_variant(arm, [row[0] for row in encoded]) for arm in MAPPINGS}
    train, test = [list(corpus(archive, split)) for split in ("dev", "Gold_test")]
    trial_ids = [{paper for _, _, _, _, sources, _ in cases for paper, _ in sources} for cases in (train,test)]
    if len(train) != 200 or len(test) != 500 or trial_ids[0] & trial_ids[1]:
        raise ValueError("Unexpected calibration or test trial inventory")
    links = [dict(case=row[0], doc=paper) for row in test for paper, _ in row[4]]
    group_by_case = {link["case"]: group for link, group in zip(links, components(links), strict=True)}
    groups = [group_by_case[row[0]] for row in test]

    def collect(cases, view, arms):
        rows = []
        for pos, (key, query, section, label, sources, gold) in enumerate(cases):
            selected = choice(key, query, sources, controller, cutoff, "fixed_two") if view == "fixed_two" else gold
            offset = pos % len(arms)
            readings = {}
            for arm in arms[offset:] + arms[:offset]:
                prompt, digits = prompts[arm]
                start = perf_counter()
                logits, tokens = label_scores(tokenizer, reader, digits, query, section, sources, selected, prompt=prompt)
                readings[arm] = dict(margin=float(logits[0] - logits[1]), tokens=tokens,
                                     ms=(perf_counter() - start) * 1000)
            rows.append(dict(target=label == "Entailment", readings=readings))
            if (pos + 1) % 100 == 0:
                print(view, len(arms), pos + 1, flush=True)
        return rows

    report = {}
    for view in VIEWS:
        calibration = collect(train, view, MAPPINGS[:2])
        correction = mean((r["readings"]["swap_tokens"]["margin"] - r["readings"]["original"]["margin"]) / 2
                          for r in calibration)
        rows = collect(test, view, MAPPINGS)
        report[view] = dict(correction=correction, results=summarize(rows, correction, groups))
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            reader_revision=provenance["revision"], test_cases=len(test), calibration_cases=len(train),
                            archive_sha256=ARCHIVE, frozen_controller_sha256=digest(frozen),
                            script_sha256=digest(Path(__file__)),
                            reader_script_sha256=digest(Path(label_scores.__code__.co_filename)), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "frozen", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
