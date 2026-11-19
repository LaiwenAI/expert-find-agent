"""Compare compact slow-reader views with full contexts and live selective verification."""

import argparse
from collections import Counter
import os
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.qasper.paired_reader import load_reader
from experiments.scifact.criterion_reader import measure as read_strong
from experiments.scifact.criterion_verifier_calibration import ARCHIVE, CAP, calibrate, counts, prepare, read_jsonl
from experiments.scifact.nli_evidence_windows import choose_windows
from experiments.scifact.selective_evidence_verifier import FALLBACK_CAP, as_margins, choose_gate, support_margin
from experiments.scifact.small_nli_verifier import BATCH, load, measure as read_short


DIRECT = ("full", "lead_three", "lexical_three", "two_lexical_windows")
FALLBACK = {"selective_full": "full", "selective_three": "lexical_three"}
ARMS = (*DIRECT, *FALLBACK)


def source_view(rows, abstracts, mode):
    if mode == "full":
        return rows
    selected = []
    for row in rows:
        sentences = abstracts[row["doc"]]
        if mode == "lead_three":
            indices = range(min(3, len(sentences)))
        else:
            first, two = choose_windows(row["criterion"], sentences)
            indices = first if mode == "lexical_three" else two
        selected.append(row | {"source": " ".join(sentences[i] for i in indices)})
    return selected


def combine(block, abstracts, fast, strong, cutoffs, gates, arm):
    start = perf_counter()
    if arm in DIRECT:
        values, _ = read_strong(*strong, source_view(block, abstracts, arm))
        decisions = support_margin(values) > cutoffs[arm]
        sent = len(block)
    else:
        compact = source_view(block, abstracts, "lexical_three")
        verdicts, _ = read_short(*fast, compact)
        grades = as_margins(verdicts, 0.)
        ok = np.array([row is not None for row in verdicts])
        gate = gates[arm]
        decisions = ok & (grades > gate["accept"])
        pending = ~ok | ((grades > gate["reject"]) & (grades <= gate["accept"]))
        selected = np.flatnonzero(pending)
        sent = len(selected)
        if sent:
            mode = FALLBACK[arm]
            excerpt = [compact[i] if mode == "lexical_three" else block[i] for i in selected]
            diagnoses, _ = read_strong(*strong, excerpt)
            decisions[selected] = support_margin(diagnoses) > cutoffs[mode]
    return decisions.tolist(), sent, (perf_counter() - start) * 1000


def benchmark(test, abstracts, fast, strong, cutoffs, gates, width):
    trials = {arm: {"accepted": [], "seconds": [], "strong": 0} for arm in ARMS}
    for i, offset in enumerate(range(0, len(test), width)):
        rows = test[offset:offset+width]
        start = i % len(ARMS)
        for arm in ARMS[start:] + ARMS[:start]:
            accepted, count, ms = combine(rows, abstracts, fast, strong, cutoffs, gates, arm)
            stats = trials[arm]
            stats["accepted"].extend(accepted)
            stats["seconds"].append(ms)
            stats["strong"] += count
    expected = [row["label"] for row in test]
    return {
        arm: dict(labels=counts(row["accepted"], expected), full_model_inputs=row["strong"],
                  p50_p95_ms=np.quantile(row["seconds"], [.5, .95]).tolist())
        for arm, row in trials.items()
    }


def run(archive_path, short_checkpoint, full_checkpoint, output):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output on a Slurm GPU")
    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        abstracts = {str(row["doc_id"]): row["abstract"] for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    texts = {key: " ".join(values) for key, values in abstracts.items() if values}
    held = {str(doc) for row in dev for doc in (*row["cited_doc_ids"], *row["evidence"])}
    controls, test = prepare(train, texts, held), prepare(dev, texts, set())
    if (len(controls), len(test)) != (309, 226):
        raise ValueError("Changed comparison cohort")
    short_tok, short, idx, fast_info = load(short_checkpoint, "mdeberta")
    full_tok, full, strong_info = load_reader(full_checkpoint)
    fast, strong = (short_tok, short, idx), (full_tok, full)
    tags = [row["label"] for row in controls]
    grades, _ = read_short(*fast, source_view(controls, abstracts, "lexical_three"))
    if any(row is None for row in grades):
        raise ValueError("The calibration short route abstained on a token limit")
    margins = {}
    for arm in DIRECT:
        values, _ = read_strong(*strong, source_view(controls, abstracts, arm))
        margins[arm] = support_margin(values)
    cutoffs = {arm: float(calibrate(values.tolist(), tags)[0]) for arm, values in margins.items()}
    grades = as_margins(grades, 0.)
    gates = {arm: choose_gate(grades, margins[mode], tags, cutoffs[mode]) for arm, mode in FALLBACK.items()}
    print("Frozen cutoffs", cutoffs, "and gates", gates, flush=True)
    for arm in ARMS:
        combine(controls[:BATCH], abstracts, fast, strong, cutoffs, gates, arm)
    batch = benchmark(test, abstracts, fast, strong, cutoffs, gates, BATCH)
    serial = benchmark(test, abstracts, fast, strong, cutoffs, gates, 1)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
        models={"short": fast_info["revision"], "strong": strong_info["revision"]},
        calibration_cases=len(controls), test_cases=len(test), test_labels=dict(Counter(row["label"] for row in test)),
        calibration_false_support_cap=CAP, calibration_fallback_cap=FALLBACK_CAP,
        cutoffs=cutoffs, gates=gates, batch=batch, serial=serial,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive-path", "short-checkpoint", "full-checkpoint", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    run(**vars(parser.parse_args()))
