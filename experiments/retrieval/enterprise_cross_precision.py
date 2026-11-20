"""Measure selective cross-precision rechecks on the enterprise parser."""

import argparse
from collections import Counter
from pathlib import Path
import json
import os
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_guarded_parsing import GuardedReader, recheck_margin, target
from experiments.retrieval.enterprise_isolated_parsing import blocks, shared_prompt, spans
from experiments.retrieval.enterprise_precision_guard import (
    DEVELOPMENT, QUALIFICATION, evaluate as baseline,
)


ARMS = ("bf16_independent", "bf16_guard", "fp16_guard", "bf16_narrow", "consensus")
WIDE, NARROW = 2., .5
CONTROLS = "ca96d57855c1acb463a3bbce8bb21e3578d5c8465c340958fd74fe209cea5000"


def prefix_cache(reader, front):
    cache = reader.backbone(input_ids=torch.tensor([front], device="cuda"),
                            use_cache=True).past_key_values
    if cache.get_seq_length() != len(front):
        raise ValueError("Incomplete prefix cache")
    return cache


def qualification_gap(logits):
    return recheck_margin(logits, "qualification_recheck")


def disagreements(first, second):
    a = first.argmax(dim=1) == QUALIFICATION
    b = second.argmax(dim=1) == QUALIFICATION
    return bool((a != b).any().item())


def evaluate(text, clauses, readers, arm, *, prefill=prefix_cache, recheck=None):
    if arm not in ("bf16_narrow", "consensus"):
        return baseline(text, clauses, readers, arm)
    started = perf_counter()
    bf16, fp16 = readers["bf16"], readers["fp16"]
    front, suffixes = shared_prompt(bf16.tokenizer, text, clauses)
    cache = prefill(bf16, front)
    secondary = None
    extra = Counter()
    original_slots = len(front)
    indices, answers = [], []
    for block in blocks(front, suffixes):
        coordinates, items = zip(*block, strict=True)
        scores, slots = bf16.cached(front, items, cache)
        original_slots += slots
        needs_fallback = qualification_gap(scores) <= (WIDE if arm == "consensus" else NARROW)
        alt = None
        if needs_fallback and arm == "consensus":
            if secondary is None:
                secondary = prefill(fp16, front)
                extra["fp16_prefixes"] += 1
                extra["fp16_slots"] += len(front)
            alt, slots = fp16.cached(front, items, secondary)
            extra["fp16_block_checks"] += 1
            extra["fp16_slots"] += slots
            diverges = disagreements(scores, alt)
            ambiguous = qualification_gap(alt) <= NARROW
            extra["disagreeing_blocks"] += diverges
            extra["uncertain_fp16_blocks"] += ambiguous
            needs_fallback = diverges or ambiguous
        if needs_fallback:
            if recheck is None:
                scores, slots = bf16.independent(front, items)
            else:
                scores, slots = recheck(bf16, front, items, scores, alt)
            extra["rechecked"] += 1
            extra["fallback_slots"] += slots
        indices.extend(coordinates)
        answers.append(scores.float())
    logits = np.empty((len(clauses), 4), dtype=np.float32)
    logits[indices] = torch.cat(answers).cpu().numpy()
    elapsed = (perf_counter() - started)*1000
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite precision-cascade scores")
    return dict(logits=logits, labels=logits.argmax(axis=1),
                original_slots=original_slots, ms=elapsed) | dict(extra)


def summarize(records):
    results = {}
    rng = np.random.default_rng(20261119)
    draws = rng.integers(len(records), size=(4000, len(records)))
    guard_ms = np.array([row["bf16_guard"]["ms"] for row, _, _ in records])
    recheck_fields = ("rechecked", "original_slots", "fallback_slots", "fp16_prefixes",
                      "fp16_block_checks", "fp16_slots", "disagreeing_blocks", "uncertain_fp16_blocks")
    for arm in ARMS:
        counts = Counter()
        for row, idx, gold in records:
            reference, result = row["bf16_independent"]["labels"], row[arm]
            actual = result["labels"]
            target_qual, actual_qual = reference == QUALIFICATION, actual == QUALIFICATION
            counts["clauses"] += len(reference)
            counts["role_changes"] += int(np.count_nonzero(actual != reference))
            counts["qualification_requests_changed"] += int(np.any(actual_qual != target_qual))
            counts["qualifications_added"] += int(np.count_nonzero(actual_qual & ~target_qual))
            counts["qualifications_removed"] += int(np.count_nonzero(~actual_qual & target_qual))
            if idx is not None:
                counts["targets"] += 1
                counts["targets_correct"] += int(actual[idx] == gold)
                counts["false_qualification_targets"] += int(actual_qual[idx] and gold != QUALIFICATION)
                counts["missed_qualification_targets"] += int(not actual_qual[idx] and gold == QUALIFICATION)
            for field in recheck_fields:
                counts[field] += result.get(field, 0)
        ms = np.asarray([row[arm]["ms"] for row, _, _ in records])
        speedup = np.quantile(guard_ms[draws], .95, axis=1) / np.quantile(ms[draws], .95, axis=1)
        results[arm] = dict(counts, p50_p95_ms=np.quantile(ms, [.5,.95]).tolist(),
                            guard_over_arm_p95_ci95=np.quantile(speedup, [.025,.975]).tolist())
    return results


@torch.inference_mode()
def run(development, controls, checkpoint, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or digest(development) != DEVELOPMENT or digest(controls) != CONTROLS):
        raise ValueError("Require pinned cohorts and a fresh Slurm allocation")
    public = json.loads(controls.read_text())
    real = list(map(json.loads, development.read_text().splitlines()))
    if len(public) != 72 or len(real) != 80 or any(row["split"] != "development" for row in real):
        raise ValueError("Unexpected cohort")
    torch.backends.cuda.matmul.allow_tf32 = False
    readers = {kind: GuardedReader(checkpoint) for kind in ("bf16", "fp16")}
    half = readers["fp16"]
    half.model.half()
    half.head = half.model.get_output_embeddings().weight[half.labels].contiguous()
    if half.head.dtype != torch.float16:
        raise ValueError("Incorrect half-precision reader")
    warmup = public[0]["text"]
    for arm in ARMS:
        evaluate(warmup, spans(warmup), readers, arm)
    report = {}
    for name, dataset in (("controls", public), ("development", real)):
        records = []
        for i, row in enumerate(dataset):
            text = row["text"]
            clauses = spans(text)
            offset = i % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            outputs = {arm: evaluate(text, clauses, readers, arm) for arm in order}
            records.append((outputs, *target(text, clauses, row)))
            if (i + 1) % 20 == 0:
                print(name, i+1, flush=True)
        report[name] = dict(requests=len(records), arms=summarize(records))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), development_sha256=DEVELOPMENT,
        controls_sha256=CONTROLS, script_sha256=digest(Path(__file__)),
        model_revision=readers["bf16"].provenance["revision"], trigger_margins=dict(wide=WIDE, narrow=NARROW),
        results=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
