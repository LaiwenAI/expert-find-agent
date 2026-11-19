"""Measure the live quality and batch latency of selective evidence verification."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.qasper.paired_reader import load_reader
from experiments.scifact.criterion_reader import measure as read_full
from experiments.scifact.criterion_verifier_calibration import ARCHIVE, CAP, calibrate, counts, prepare, read_jsonl, row_key
from experiments.scifact.nli_evidence_windows import choose_windows
from experiments.scifact.small_nli_verifier import BATCH, load, measure as read_short


FALLBACK_CAP = 0.50
ARMS = ("short", "full", "fixed_sample", "selective")


def windows(rows, abstracts):
    output = []
    for row in rows:
        text = abstracts[row["doc"]]
        first, _ = choose_windows(row["criterion"], text)
        output.append(row | {"source": " ".join(text[i] for i in first)})
    return output


def support_margin(scores):
    return scores[:, 0] - scores[:, 1:].max(axis=1)


def as_margins(values, outside):
    return np.array([r[1] if r is not None else outside for r in values])


def choose_gate(short, full, gold, cutoff, fixed_pick=None):
    edges = np.r_[np.nextafter(min(short), -np.inf), np.unique(np.quantile(short, np.linspace(0., 1., 21)))]
    full_positive = full > cutoff
    best = None
    for reject in edges:
        for accept in edges[edges >= reject]:
            in_band = (short > reject) & (short <= accept)
            use = in_band if fixed_pick is None else in_band & fixed_pick
            if use.mean() > FALLBACK_CAP:
                continue
            decisions = (short > accept) | (use & full_positive)
            report = counts(decisions.tolist(), gold)
            negatives = [report[name] for name in ("contradict", "unknown")]
            if any(row["predicted_support"] > CAP * row["cases"] for row in negatives):
                continue
            merit = (report["support"]["predicted_support"],
                     -sum(row["predicted_support"] for row in negatives), -int(use.sum()))
            if best is None or merit > best[0]:
                best = merit, dict(reject=float(reject), accept=float(accept), calibration_report=report,
                                   calibration_fallbacks=int(use.sum()))
    if best is None:
        raise ValueError("No gate met the development limits")
    return best[1]


def choose_fixed(rows):
    return np.array([int.from_bytes(row_key(row["case"], row["doc"])[:8], "big") < (1 << 63) for row in rows])


def assess(rows, abstracts, fast, strong, thresholds, gates, arm):
    start = perf_counter()
    calls = 0
    if arm == "full":
        values, _ = read_full(*strong, rows)
        decisions = support_margin(values) > thresholds["full"]
        calls = len(rows)
    else:
        short, _ = read_short(*fast, windows(rows, abstracts))
        # An out-of-context fast input is undecidable and can only escalate.
        values = as_margins(short, 0.)
        valid = np.array([row is not None for row in short])
        if arm == "short":
            decisions = valid & (values > thresholds["short"])
        else:
            rule = gates[arm]
            decisions = valid & (values > rule["accept"])
            uncertain = ~valid | ((values > rule["reject"]) & (values <= rule["accept"]))
            if arm == "fixed_sample":
                uncertain &= choose_fixed(rows)
            chosen = np.flatnonzero(uncertain)
            calls = len(chosen)
            if chosen.size:
                read, _ = read_full(*strong, [rows[i] for i in chosen])
                decisions[chosen] = support_margin(read) > thresholds["full"]
    return decisions.tolist(), calls, (perf_counter() - start) * 1000


def evaluate_live(test, abstracts, fast, strong, thresholds, gates, batch_size):
    results = {arm: {"decisions": [], "calls": [], "times": []} for arm in ARMS}
    for i, offset in enumerate(range(0, len(test), batch_size)):
        block = test[offset:offset + batch_size]
        rotation = i % len(ARMS)
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            decisions, calls, elapsed = assess(block, abstracts, fast, strong, thresholds, gates, arm)
            results[arm]["decisions"].extend(decisions)
            results[arm]["calls"].append(calls)
            results[arm]["times"].append(elapsed)
    gold = [row["label"] for row in test]
    report = {}
    for arm, row in results.items():
        report[arm] = dict(
            labels=counts(row["decisions"], gold), full_reader_inputs=sum(row["calls"]),
            p50_p95_ms=np.quantile(row["times"], [.5, .95]).tolist(),
            full_reader_batches=sum(n > 0 for n in row["calls"]),
        )
    return report


def run(archive_path, short_checkpoint, full_checkpoint, output):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run on a Slurm GPU allocation with a new output path")
    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        abstracts = {str(row["doc_id"]): row["abstract"] for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    texts = {doc: " ".join(sents) for doc, sents in abstracts.items() if sents}
    covered = {str(doc) for row in dev for doc in (*row["cited_doc_ids"], *row["evidence"])}
    fit, test = prepare(train, texts, covered), prepare(dev, texts, set())
    if (len(fit), len(test)) != (309, 226):
        raise ValueError("Changed comparison cohort")
    tokenizer, short, idx, short_provenance = load(short_checkpoint, "mdeberta")
    reader_tokenizer, full, full_provenance = load_reader(full_checkpoint)
    fast, strong = (tokenizer, short, idx), (reader_tokenizer, full)
    gold_fit = [row["label"] for row in fit]
    fast_control, _ = read_short(*fast, windows(fit, abstracts))
    if any(row is None for row in fast_control):
        raise ValueError("A calibration short window exceeds its context")
    fast_margin = as_margins(fast_control, 0.)
    reader_control, _ = read_full(*strong, fit)
    full_margin = support_margin(reader_control)
    thresholds = {}
    for name, scores in (("short", fast_margin), ("full", full_margin)):
        thresholds[name] = float(calibrate(scores.tolist(), gold_fit)[0])
    gates = {
        "fixed_sample": choose_gate(fast_margin, full_margin, gold_fit, thresholds["full"], choose_fixed(fit)),
        "selective": choose_gate(fast_margin, full_margin, gold_fit, thresholds["full"]),
    }
    print("Froze gates", gates, flush=True)
    for arm in ARMS:
        assess(fit[:BATCH], abstracts, fast, strong, thresholds, gates, arm)
    report = evaluate_live(test, abstracts, fast, strong, thresholds, gates, BATCH)
    serial_report = evaluate_live(test, abstracts, fast, strong, thresholds, gates, 1)
    gold = [row["label"] for row in test]
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
        models={"short": short_provenance["revision"], "full": full_provenance["revision"]},
        train_cases=len(fit), test_cases=len(test), label_totals=dict(Counter(gold)),
        calibration_false_support_cap=CAP, calibration_fallback_cap=FALLBACK_CAP,
        thresholds=thresholds, gates=gates, report=report, serial_report=serial_report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive-path", "short-checkpoint", "full-checkpoint", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(**vars(parser.parse_args()))
