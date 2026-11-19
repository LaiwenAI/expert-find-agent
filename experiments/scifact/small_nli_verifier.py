"""Measure two small multilingual NLI models on the frozen SciFact verifier cohort."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.followir.core17_reranking import digest, write_json


REVISIONS = {
    "mdeberta": ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"),
    "minilm": ("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "0a71e92a985b6e1ad1828cf67ce9c459639c1dca"),
}
LABELS = ("entailment", "contradiction", "neutral")
BATCH = 16
MAX_TOKENS = 512


def load(checkpoint, model_name):
    source = json.loads((checkpoint / "local-provenance.json").read_text())
    if (source["repo"], source["revision"]) != REVISIONS[model_name]:
        raise ValueError("Unexpected NLI checkpoint")
    for name, expected in source["files"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Changed NLI checkpoint file: {name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    reader = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    label_index = {label.casefold(): int(index) for index, label in reader.config.id2label.items()}
    if set(label_index) != set(LABELS):
        raise ValueError("Unexpected entailment label mapping")
    return tokenizer, reader, [label_index[name] for name in LABELS], source


def measure(tokenizer, reader, positions, rows, *, batch_size=BATCH):
    verdicts = []
    times = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        started = perf_counter()
        encoded = tokenizer([row["source"] for row in batch], [row["criterion"] for row in batch],
                            truncation=False, padding=False, verbose=False)
        selected = [i for i, ids in enumerate(encoded["input_ids"]) if len(ids) <= MAX_TOKENS]
        choices = [None] * len(batch)
        if selected:
            padded = tokenizer.pad([{key: vals[i] for key, vals in encoded.items()} for i in selected],
                                   padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = reader(**padded).logits[:, positions].float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite scores")
            for i, logit in zip(selected, logits, strict=True):
                label = LABELS[int(logit.argmax())]
                margin = float(logit[0] - max(logit[1:]))
                choices[i] = (label, margin)
        times.append((perf_counter() - started) * 1000)
        verdicts.extend(choices)
    return verdicts, times


def fit(scores, gold):
    from experiments.scifact.criterion_verifier_calibration import calibrate

    valid = [(score[1], label) for score, label in zip(scores, gold, strict=True) if score is not None]
    margins = [m for m, _ in valid]
    labels = [label for _, label in valid]
    if set(labels) != set(gold):
        raise ValueError("No valid calibration inputs for a reference class")
    threshold, metrics = calibrate(margins, labels)
    return threshold, metrics


def summarize(scores, gold, threshold):
    from experiments.scifact.criterion_verifier_calibration import counts

    return dict(
        raw_support=counts([row is not None and row[0] == "entailment" for row in scores], gold),
        calibrated_support=counts([row is not None and row[1] > threshold for row in scores], gold),
        skipped=dict(Counter(label for row, label in zip(scores, gold, strict=True) if row is None)),
        confusion=[dict(gold=g, predicted=p, cases=n) for (g, p), n in sorted(Counter(
            (label, row[0] if row is not None else "over_limit")
            for row, label in zip(scores, gold, strict=True)
        ).items())],
    )


def run(archive_path, checkpoints, output, private_input=None, private_output=None):
    from experiments.scifact.criterion_verifier_calibration import ARCHIVE, CAP, prepare, read_jsonl

    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or (private_input is None) != (private_output is None)
            or (private_output is not None and private_output.exists())):
        raise ValueError("Require fresh result paths and a Slurm GPU")
    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        docs = {str(row["doc_id"]): " ".join(row["abstract"]) for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    blocked = {str(doc) for c in dev for doc in (*c["cited_doc_ids"], *c["evidence"])}
    calibration = prepare(train, docs, blocked)
    test = prepare(dev, docs, set())
    gold_calibration = [row["label"] for row in calibration]
    gold_test = [row["label"] for row in test]
    private = []
    if private_input is not None:
        private = [json.loads(line) for line in private_input.read_text().splitlines()]
        if len({row["case_id"] for row in private}) != len(private):
            raise ValueError("Duplicate private review case")
    private_verdicts = {row["case_id"]: {} for row in private}
    report = {}
    for name in REVISIONS:
        tokenizer, reader, positions, provenance = load(checkpoints / f"{name}-nli", name)
        measure(tokenizer, reader, positions, calibration[:BATCH])
        fitted, _ = measure(tokenizer, reader, positions, calibration)
        threshold, calibration_counts = fit(fitted, gold_calibration)
        tested, times = measure(tokenizer, reader, positions, test)
        timing = np.quantile(times, [.5, .95]).tolist()
        report[name] = dict(
            repo=provenance["repo"], revision=provenance["revision"], threshold=threshold,
            calibration_no_truncation=calibration_counts,
            evaluation=summarize(tested, gold_test, threshold),
            evaluation_batch_p50_p95_ms=timing,
        )
        if private:
            results, _ = measure(tokenizer, reader, positions, private)
            for row, verdict in zip(private, results, strict=True):
                private_verdicts[row["case_id"]][name] = None if verdict is None else {
                    "verdict": verdict[0], "calibrated_support": verdict[1] > threshold}
        del reader
        print(name, "finished", flush=True)
    if private_output is not None:
        with private_output.open("x", encoding="utf-8") as stream:
            private_output.chmod(0o600)
            for key, verdict in private_verdicts.items():
                stream.write(json.dumps(dict(case_id=key, verdicts=verdict)) + "\n")
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)), max_tokens=MAX_TOKENS,
        calibration_cases=len(calibration), evaluation_cases=len(test), threshold_cap=CAP, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-path", "checkpoints", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("private-input", "private-output"):
        parser.add_argument(f"--{name}", type=Path)
    run(**vars(parser.parse_args()))
