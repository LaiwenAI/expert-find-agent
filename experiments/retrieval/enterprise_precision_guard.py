"""Compare cached FP16 and BF16 qualification classification on a frozen panel."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_guarded_parsing import GuardedReader, recheck_margin, target
from experiments.retrieval.enterprise_isolated_parsing import blocks, shared_prompt, spans


DEVELOPMENT = "30661871ac08a61e7e74d67d75491878c7e4194f217483b60d06426c4b36e4c5"
ARMS = ("bf16_independent", "bf16_cache", "bf16_guard", "fp16_independent", "fp16_cache", "fp16_guard")
CUTOFFS = {"bf16_guard": 2., "fp16_guard": .5}
QUALIFICATION = 2


def evaluate(text, clauses, readers, arm):
    precision, route = arm.split("_", 1)
    reader = readers[precision]
    started = perf_counter()
    front, suffixes = shared_prompt(reader.tokenizer, text, clauses)
    coordinates, results = [], []
    rechecked = original_slots = fallback_slots = 0
    prefix = None
    if route != "independent":
        prefix = reader.backbone(input_ids=torch.tensor([front], device="cuda"),
                                 use_cache=True).past_key_values
        if prefix.get_seq_length() != len(front):
            raise ValueError("Incomplete prefix cache")
        original_slots += len(front)
    for block in blocks(front, suffixes):
        ids, items = zip(*block, strict=True)
        if route == "independent":
            logits, slots = reader.independent(front, items)
        else:
            logits, slots = reader.cached(front, items, prefix)
        original_slots += slots
        if route == "guard" and recheck_margin(logits, "qualification_recheck") <= CUTOFFS[arm]:
            logits, slots = readers["bf16"].independent(front, items)
            fallback_slots += slots
            rechecked += 1
        coordinates.extend(ids)
        results.append(logits.float())
    output = np.empty((len(clauses), 4), dtype=np.float32)
    output[coordinates] = torch.cat(results).cpu().numpy()
    elapsed = (perf_counter() - started) * 1000
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite reader scores")
    return dict(logits=output, labels=output.argmax(axis=1), rechecked=rechecked,
                original_slots=original_slots, fallback_slots=fallback_slots, ms=elapsed)


def summarize(rows):
    stats = {}
    rng = np.random.default_rng(20261119)
    indices = rng.integers(len(rows), size=(4000, len(rows)))
    guarded = np.asarray([row["bf16_guard"]["ms"] for row, _, _ in rows])
    for arm in ARMS:
        results = [row[arm] for row, _, _ in rows]
        selected = Counter()
        errors = []
        for result, (row, idx, gold) in zip(results, rows, strict=True):
            reference = row["bf16_independent"]
            same_precision = row[f"{arm.split('_')[0]}_independent"]
            labels, orig = result["labels"], reference["labels"]
            selected["clauses"] += len(orig)
            selected["changed_roles"] += int(sum(labels != orig))
            selected["changed_qualification_requests"] += not np.array_equal(labels == QUALIFICATION, orig == QUALIFICATION)
            selected["qualification_added"] += int(sum((labels == QUALIFICATION) & (orig != QUALIFICATION)))
            selected["qualification_removed"] += int(sum((orig == QUALIFICATION) & (labels != QUALIFICATION)))
            selected["same_precision_qualification_disagreements"] += int(
                sum((labels == QUALIFICATION) != (same_precision["labels"] == QUALIFICATION)))
            for field in ("rechecked", "original_slots", "fallback_slots"):
                selected[field] += result[field]
            if idx is not None:
                selected["control_targets"] += 1
                selected["correct_controls"] += labels[idx] == gold
                selected["false_qualification_controls"] += labels[idx] == QUALIFICATION and gold != QUALIFICATION
                selected["missed_qualification_controls"] += labels[idx] != QUALIFICATION and gold == QUALIFICATION
            errors.extend(np.max(np.abs(result["logits"] - same_precision["logits"]), axis=1))
        times = np.asarray([result["ms"] for result in results])
        ratios = np.quantile(guarded[indices], .95, axis=1) / np.quantile(times[indices], .95, axis=1)
        stats[arm] = dict(
            {key: int(value) for key, value in selected.items()}, p50_p95_ms=np.quantile(times, [.5, .95]).tolist(),
            p95_guarded_over_arm_ci95=np.quantile(ratios, [.025, .975]).tolist(),
            max_logit_drift_vs_same_precision=float(max(errors)),
        )
    return stats


@torch.inference_mode()
def run(development, controls, checkpoint, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or digest(development) != DEVELOPMENT):
        raise ValueError("Require the pinned development export and a new GPU run")
    real = list(map(json.loads, development.read_text().splitlines()))
    public = json.loads(controls.read_text())
    if len(real) != 80 or len(public) != 72 or any(item["split"] != "development" for item in real):
        raise ValueError("Unexpected benchmark cohorts")
    torch.backends.cuda.matmul.allow_tf32 = False
    readers = {kind: GuardedReader(checkpoint) for kind in ("bf16", "fp16")}
    half = readers["fp16"]
    half.model.half()
    half.head = half.model.get_output_embeddings().weight[half.labels].contiguous()
    if half.head.dtype != torch.float16:
        raise ValueError("Incorrect cached reader precision")
    sample = public[0]["text"]
    for arm in ARMS:
        evaluate(sample, spans(sample), readers, arm)
    report = {}
    for name, dataset in (("controls", public), ("development", real)):
        outputs = []
        for index, row in enumerate(dataset):
            text = row["text"]
            clauses = spans(text)
            offset = index % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            outputs.append(({arm: evaluate(text, clauses, readers, arm) for arm in order}, *target(text, clauses, row)))
            if (index + 1) % 20 == 0:
                print(name, index+1, flush=True)
        report[name] = dict(requests=len(outputs), arms=summarize(outputs))
    write_json(output, dict(
        development_sha256=DEVELOPMENT, controls_sha256=digest(controls), script_sha256=digest(Path(__file__)),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        model=readers["bf16"].provenance, guard_cutoffs=CUTOFFS, results=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
