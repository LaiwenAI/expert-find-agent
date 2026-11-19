"""Apply frozen VitaminC verifier weights and cutoffs to SciFact."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tarfile

import numpy as np

from expert_search.evidence_windows import choose_windows
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json

LABELS = ("support", "contradict", "unknown")
VIEWS = ("full", "lexical_three")


def load_panels(archive_path):
    from experiments.scifact.criterion_verifier_calibration import ARCHIVE, prepare, read_jsonl

    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        abstracts = {str(row["doc_id"]): row["abstract"] for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    docs = {name: " ".join(sentences) for name, sentences in abstracts.items() if sentences}
    blocked = {str(doc) for row in dev for doc in (*row["cited_doc_ids"], *row["evidence"])}
    panels = {"calibration": prepare(train, docs, blocked), "test": prepare(dev, docs, set())}
    if tuple(map(len, panels.values())) != (309, 226):
        raise ValueError("Changed SciFact evaluation cohort")
    if {x["doc"] for x in panels["calibration"]} & {x["doc"] for x in panels["test"]}:
        raise ValueError("Source leakage across calibration and evaluation")
    for name, rows in panels.items():
        claims = {r["id"]: r for r in (train if name == "calibration" else dev)}
        for row in rows:
            sentences = abstracts[row["doc"]]
            kept, _ = choose_windows(row["criterion"], sentences)
            row["lexical_three"] = " ".join(sentences[i] for i in kept)
            rationale = claims[row["case"]]["evidence"].get(row["doc"], [])
            row["complete_annotation"] = any(set(g["sentences"]) <= set(kept) for g in rationale)
    return panels


def summary(margins, gold, threshold, reference=None):
    pred = margins > threshold
    rows = {}
    if reference is not None:
        changes = pred.astype(int) - reference.astype(int)
    for label in LABELS:
        group = gold == label
        row = dict(total=int(group.sum()), accepted=int(pred[group].sum()))
        if reference is not None:
            row["delta_vs_frozen"] = paired_interval(changes[group])
        rows[label] = row
    return rows


def cutoff(scores, labels):
    # Choose the least restrictive single threshold satisfying both 5% caps.
    values = []
    for label in LABELS[1:]:
        group = scores[labels == label]
        cap = math.floor(.05 * len(group))
        values.append(np.sort(group)[::-1][cap])
    threshold = float(max(values))
    if not math.isfinite(threshold):
        raise ValueError("Calibration needs enough finite negatives")
    return threshold


def apply_weights(model, path, expected):
    import torch
    from safetensors.torch import load_file

    if digest(path) != expected:
        raise ValueError(f"Unverified saved weights: {path.name}")
    values = load_file(str(path), device="cpu")
    parameters = dict(model.named_parameters())
    last = model.config.num_hidden_layers - 1
    names = {name for name in parameters if name.startswith(("classifier.", f"roberta.encoder.layer.{last}."))}
    if values.keys() != names:
        raise ValueError("Expected the complete saved parameter subset")
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(values[name])


def run(archive_path, training, weights, checkpoint, output):
    import torch
    from experiments.scifact.small_nli_verifier import load, measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result file and a Slurm CUDA allocation")
    panels = load_panels(archive_path)
    trained = json.loads(training.read_bytes())
    expected = {"frozen", *(f"{a}_{seed}" for a in ("classification", "contrastive", "risk_tail")
                             for seed in (20261119, 20261120))}
    if trained["report"].keys() != expected:
        raise ValueError("Unexpected training arms")
    records = {}
    labels = {name: np.array([r["label"] for r in rows]) for name, rows in panels.items()}
    for arm, train in trained["report"].items():
        tokenizer, model, positions, provenance = load(checkpoint, "minilm")
        if trained["model_revision"] != provenance["revision"]:
            raise ValueError("Changed MiniLM checkpoint")
        if arm != "frozen":
            apply_weights(model, weights/f"{arm}.safetensors", train["training"]["weight_sha256"])
        measure(tokenizer, model, positions, panels["test"][:4], batch_size=1)
        records[arm] = {}
        for view in VIEWS:
            results = {}
            for split, rows in panels.items():
                inputs = [row | {"source": row["lexical_three"]} for row in rows] if view == "lexical_three" else rows
                scores, times = measure(tokenizer, model, positions, inputs, batch_size=1)
                results[split] = np.array([x[1] if x is not None else -np.inf for x in scores])
                results[split + "_p95_ms"] = float(np.quantile(times, .95))
            records[arm][view] = results
        print(arm, "evaluated", flush=True)

    report = {}
    for arm, views in records.items():
        report[arm] = {}
        for view, result in views.items():
            control = records["frozen"][view]
            frozen_source = trained["report"]["frozen"]["operating_points"]["strict"]["threshold"]
            source = trained["report"][arm]["operating_points"]["strict"]["threshold"]
            target = cutoff(result["calibration"], labels["calibration"])
            frozen_target = cutoff(control["calibration"], labels["calibration"])
            test = result["test"]
            complete_mask = np.array([r["complete_annotation"] for r in panels["test"]]) if view == "lexical_three" else labels["test"]!="unknown"
            report[arm][view] = dict(
                p95_ms=result["test_p95_ms"], skipped=int(np.isneginf(test).sum()),
                source_threshold=source, target_threshold=target,
                source_threshold_test=summary(test, labels["test"], source, control["test"] > frozen_source),
                target_threshold_calibration=summary(result["calibration"], labels["calibration"], target),
                target_threshold_test=summary(test, labels["test"], target, control["test"] > frozen_target),
                complete_annotation_support_accepted=int(np.sum((test>target) & complete_mask & (labels["test"]=="support"))),
                complete_annotation_support_total=int(np.sum(complete_mask & (labels["test"]=="support"))),
            )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model_revision=provenance["revision"],
        archive_sha256=digest(archive_path), source_training_sha256=digest(training), script_sha256=digest(Path(__file__)),
        panels={name:dict(Counter(r["label"] for r in q)) for name,q in panels.items()}, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive_path", "training", "weights", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
