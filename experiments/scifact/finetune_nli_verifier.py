"""Fine-tune a multilingual verifier on source-disjoint SciFact support judgments."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tarfile
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import save_file

from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.criterion_verifier_calibration import ARCHIVE, CAP, prepare, read_jsonl
from experiments.scifact.nli_evidence_windows import choose_windows
from experiments.scifact.small_nli_verifier import LABELS, MAX_TOKENS, fit, load, measure, summarize
from experiments.scifact.xnli_language_crossing import PROJECTION_SHA


SEED = 719
BATCH = 8
EPOCHS = 3
ARMS = {"frozen": 0., "head": 3e-4, "last_two": 2e-5}
TASK_LABELS = {"support": "entailment", "contradict": "contradiction", "unknown": "neutral"}


def split(rows):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["label"], []).append(row)
    train, calibration = [], []
    for label, group in sorted(by_class.items()):
        ordered = sorted(group, key=lambda r: sha256(f"finetune-nli-v1:{r['doc']}".encode()).digest())
        boundary = len(ordered) // 4
        calibration.extend(ordered[:boundary])
        train.extend(ordered[boundary:])
    if {row["doc"] for row in train} & {row["doc"] for row in calibration}:
        raise ValueError("Overlap between fine-tune and calibration papers")
    return train, calibration


def lexical(row, abstracts):
    sentences = abstracts[row["doc"]]
    chosen, _ = choose_windows(row["criterion"], sentences)
    return row | {"source": " ".join(sentences[i] for i in chosen)}


def training_view(row, abstracts, claims):
    info = claims[row["case"]]["evidence"].get(row["doc"], [])
    rationales = [tuple(sorted(set(p["sentences"]))) for p in info
                  if p["label"].lower() == row["label"]]
    if not rationales:
        if row["label"] != "unknown":
            raise ValueError("Directional training label requires an annotated rationale")
        return lexical(row, abstracts)
    sentences = abstracts[row["doc"]]
    chosen = min(rationales, key=lambda r: (len(r), r))
    if not chosen or chosen[0] < 0 or chosen[-1] >= len(sentences):
        raise ValueError("Invalid training rationale")
    return row | {"source": " ".join(sentences[i] for i in chosen)}


def train(tokenizer, reader, positions, rows, arm, output_dir):
    start = perf_counter()
    if arm == "frozen":
        return {"trained": 0, "updates": 0, "training_seconds": 0.0}
    prefixes = ("classifier.", "pooler.")
    if arm == "last_two":
        layers = reader.config.num_hidden_layers
        prefixes += tuple(f"deberta.encoder.layer.{i}." for i in (layers - 2, layers - 1))
    params = {}
    for name, parameter in reader.named_parameters():
        chosen = name.startswith(prefixes)
        parameter.requires_grad_(chosen)
        if chosen:
            params[name] = parameter
    if not params or (arm == "last_two" and not any(name.startswith("deberta.encoder") for name in params)):
        raise ValueError("Failed to identify parameters")
    reader.float().train()
    optimizer = torch.optim.AdamW(params.values(), lr=ARMS[arm], weight_decay=.01)
    encoded = tokenizer([row["source"] for row in rows], [row["criterion"] for row in rows],
                        truncation=False, padding=False, verbose=False)
    if max(map(len, encoded["input_ids"])) > MAX_TOKENS:
        raise ValueError("Training example exceeds the shared context limit")
    target = [positions[LABELS.index(TASK_LABELS[row["label"]])] for row in rows]
    losses = []
    for epoch in range(EPOCHS):
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)
        for offset in range(0, len(order), BATCH):
            ids = order[offset:offset+BATCH]
            batch = tokenizer.pad([{key: values[i] for key, values in encoded.items()} for i in ids],
                                  padding=True, return_tensors="pt").to("cuda")
            labels = torch.tensor([target[i] for i in ids], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = reader(**batch, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    del optimizer
    reader.bfloat16().eval()
    seconds = perf_counter() - start
    checkpoint = output_dir / f"{arm}.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in params.items()}, str(checkpoint))
    return dict(trained=sum(value.numel() for value in params.values()), updates=len(losses),
                training_seconds=seconds, final_epoch_loss=float(np.mean(losses[-int(np.ceil(len(rows)/BATCH)):])),
                adapter_sha256=digest(checkpoint))


def run(archive_path, checkpoint, xnli_path, output, weights):
    if (output.exists() or weights.exists() or not os.environ.get("SLURM_JOB_ID")
            or not torch.cuda.is_available()):
        raise ValueError("Require new destinations and a Slurm CUDA allocation")
    if digest(archive_path) != ARCHIVE or digest(xnli_path) != PROJECTION_SHA:
        raise ValueError("Changed public input")
    with tarfile.open(archive_path, "r:gz") as archive:
        abstracts = {str(row["doc_id"]): row["abstract"] for row in read_jsonl(archive, "corpus")}
        train_claims = read_jsonl(archive, "claims_train")
        dev_claims = read_jsonl(archive, "claims_dev")
    docs = {doc: " ".join(sentences) for doc, sentences in abstracts.items() if sentences}
    blocked = {str(doc) for c in dev_claims for doc in (*c["cited_doc_ids"], *c["evidence"])}
    train_rows, controls = split(prepare(train_claims, docs, blocked))
    test = prepare(dev_claims, docs, set())
    claims = {row["id"]: row for row in train_claims}
    training = [training_view(row, abstracts, claims) for row in train_rows]
    calibration = [lexical(row, abstracts) for row in controls]
    test_windows = [lexical(row, abstracts) for row in test]
    xnli = [json.loads(line) for line in xnli_path.read_text().splitlines()]
    labels = [row["label"] for row in xnli]
    weights.mkdir(mode=0o700)
    report = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        tokenizer, reader, positions, provenance = load(checkpoint, "mdeberta")
        trained = train(tokenizer, reader, positions, training, arm, weights)
        measure(tokenizer, reader, positions, calibration[:16])
        fitted, _ = measure(tokenizer, reader, positions, calibration)
        threshold, fitted_counts = fit(fitted, [row["label"] for row in calibration])
        evaluated, timings = measure(tokenizer, reader, positions, test_windows)
        language = {}
        for tag in ("en", "zh"):
            rows = [dict(source=row["premise"][tag], criterion=row["hypothesis"][tag]) for row in xnli]
            scores, _ = measure(tokenizer, reader, positions, rows)
            language[tag] = dict(total=len(rows), correct=sum(s is not None and s[0] == label
                                for s, label in zip(scores, labels, strict=True)))
        report[arm] = dict(
            train=trained, threshold=threshold, calibration=fitted_counts,
            scifact=summarize(evaluated, [row["label"] for row in test], threshold), xnli=language,
            batch_p50_p95_ms=np.quantile(timings, [.5, .95]).tolist(),
        )
        del reader
        print(arm, "complete", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        archive_sha256=ARCHIVE, xnli_sha256=PROJECTION_SHA, script_sha256=digest(Path(__file__)),
        model_revision=provenance["revision"], epochs=EPOCHS, batch=BATCH, seed=SEED, threshold_cap=CAP,
        split={name: dict(Counter(row["label"] for row in subset))
               for name, subset in (("fit", training), ("calibration", calibration), ("test", test))},
        report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-path", "checkpoint", "xnli-path", "output", "weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(**vars(parser.parse_args()))
