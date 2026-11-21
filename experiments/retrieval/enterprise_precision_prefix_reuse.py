"""Reuse a BF16 prefix when screening an FP16 qualification recheck."""

import argparse
from collections import Counter
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from experiments.retrieval.enterprise_cross_precision import (
    CONTROLS, NARROW, QUALIFICATION, WIDE, disagreements, evaluate, prefix_cache, qualification_gap,
)
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import blocks, shared_prompt, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_recheck_stress import extended_text, sample_controls
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.enterprise_shared_scope_student import timed

ARMS = ("native", "cast", "guarded_cast")
CAST_MARGIN = 1.5


def convert(cache):
    if type(cache) is not DynamicCache or cache.offloading:
        raise ValueError("Expected a resident dynamic prefix")
    result = copy(cache)
    result.layers = [copy(layer) for layer in cache.layers]
    for layer in result.layers:
        if type(layer) is not DynamicLayer or not layer.is_initialized:
            raise ValueError("Expected initialized full-attention layers")
        layer.keys = layer.keys.to(torch.float16)
        layer.values = layer.values.to(torch.float16)
    return result


def fallback(first, second):
    return bool(disagreements(first, second) or qualification_gap(second) <= NARROW)


def trust(first, second):
    return bool(torch.isfinite(second).all().item()) and not disagreements(first, second) and (
        qualification_gap(second) > CAST_MARGIN)


def reused(text, clauses, readers, mode):
    bf16, fp16 = readers["bf16"], readers["fp16"]
    front, suffixes = shared_prompt(bf16.tokenizer, text, clauses)
    primary = prefix_cache(bf16, front)
    cast = native = None
    coordinates, outputs = [], []
    work = Counter()
    for block in blocks(front, suffixes):
        indices, items = zip(*block, strict=True)
        first, _ = bf16.cached(front, items, primary)
        use_full = False
        if qualification_gap(first) <= WIDE:
            if native is None:
                if cast is None:
                    cast = convert(primary)
                    work["cast_prefixes"] += 1
                second, _ = fp16.cached(front, items, cast)
                work["cast_blocks"] += 1
                if mode == "guarded_cast" and not trust(first, second):
                    native = prefix_cache(fp16, front)
                    cast = None
                    work["fp16_prefixes"] += 1
            if native is not None:
                second, _ = fp16.cached(front, items, native)
                work["native_blocks"] += 1
            use_full = fallback(first, second)
        if use_full:
            first, slots = bf16.independent(front, items)
            work.update(rechecked=1, fallback_slots=slots)
        coordinates.extend(indices)
        outputs.append(first.float())
    labels = np.empty(len(clauses), dtype=np.int64)
    labels[coordinates] = torch.cat(outputs).argmax(dim=1).cpu().numpy()
    return labels, work


def parser_run(text, clauses, readers, mode):
    if mode != "native":
        return reused(text, clauses, readers, mode)
    output = evaluate(text, clauses, readers, "consensus")
    keys = ("fp16_prefixes", "fp16_block_checks", "rechecked", "fallback_slots")
    work = Counter({key: output.get(key, 0) for key in keys})
    work["native_blocks"] = work.pop("fp16_block_checks")
    return output["labels"], work


def request_panel(dataset, readers, reverse, public=False):
    counters = {mode: Counter() for mode in ARMS}
    latency = {mode: [] for mode in ARMS}
    trace = []
    modes = ARMS[::-1] if reverse else ARMS
    order = reversed(list(enumerate(dataset))) if reverse else enumerate(dataset)
    for index, row in order:
        text = row["text"]
        clauses = spans(text)
        position, gold = target(text, clauses, row) if public else (None, None)
        outcomes = {}
        offset = index % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            value, ms = timed(lambda: parser_run(text, clauses, readers, mode))
            outcomes[mode] = value
            latency[mode].append(ms)
        reference = outcomes["native"][0]
        signature = {}
        for mode, (labels, work) in outcomes.items():
            changed = (reference == QUALIFICATION) != (labels == QUALIFICATION)
            counters[mode].update(work)
            counters[mode].update(changed_requests=bool(changed.any()), changed_clauses=int(changed.sum()),
                                  changed_roles=int((labels != reference).sum()))
            if position is not None:
                counters[mode]["public_targets_correct"] += int(labels[position] == gold)
            signature[mode] = labels.tolist()
        trace.append((index, signature))

    distributions = {mode: np.quantile(values, [.5, .95]).tolist() for mode, values in latency.items()}
    draw = np.random.default_rng(887).integers(len(dataset), size=(4000, len(dataset)))
    reference = np.quantile(np.asarray(latency["native"])[draw], .95, axis=1)
    paired = {mode: np.quantile(np.quantile(np.asarray(values)[draw], .95, axis=1) - reference,
                               [.025, .975]).tolist() for mode, values in latency.items()}

    return dict(requests=len(dataset), counts=counters, p50_p95_ms=distributions,
                paired_p95_delta_ms_ci=paired,
                decision_signature=sha256(json.dumps(sorted(trace), sort_keys=True).encode()).hexdigest())


def forced_blocks(public, readers):
    counts = Counter()
    bf16, fp16 = readers["bf16"], readers["fp16"]
    for row in sample_controls(public):
        text = extended_text(row, 8)
        clauses = spans(text)
        position, gold = target(text, clauses, row)
        front, suffixes = shared_prompt(bf16.tokenizer, text, clauses)
        selected = next(block for block in blocks(front, suffixes) if position in dict(block))
        indices, items = zip(*selected, strict=True)
        cache = prefix_cache(bf16, front)
        primary, _ = bf16.cached(front, items, cache)
        original, _ = fp16.cached(front, items, prefix_cache(fp16, front))
        reused, _ = fp16.cached(front, items, convert(cache))
        original_fallback = fallback(primary, original)
        cast_fallback = fallback(primary, reused)
        cast_accepted = trust(primary, reused)
        canonical = bf16.independent(front, items)[0] if original_fallback or cast_fallback else None
        target_row = indices.index(position)
        baseline = (canonical if original_fallback else primary).argmax(1)
        cast = (canonical if cast_fallback else primary).argmax(1)
        guarded = primary.argmax(1) if cast_accepted else baseline
        counts.update(blocks=1, naturally_ambiguous=qualification_gap(primary) <= WIDE,
                      native_fallbacks=original_fallback, cast_fallbacks=cast_fallback,
                      trust_cast=cast_accepted, virtual_gate_disagreements=original_fallback != cast_fallback,
                      native_fallback_missed_by_guard=cast_accepted and original_fallback,
                      cast_qualification_block_changes=bool(((cast == QUALIFICATION) !=
                                                              (baseline == QUALIFICATION)).any().item()),
                      guarded_qualification_block_changes=bool(((guarded == QUALIFICATION) !=
                                                                 (baseline == QUALIFICATION)).any().item()),
                      native_target_correct=bool(baseline[target_row] == gold),
                      cast_target_correct=bool(cast[target_row] == gold),
                      guarded_target_correct=bool(guarded[target_row] == gold))
    return counts


@torch.inference_mode()
def run(args):
    require_cuda(args.output)
    if digest(args.development) != DEVELOPMENT or digest(args.controls) != CONTROLS:
        raise ValueError("Require pinned development and controls")
    public = json.loads(args.controls.read_bytes())
    development = rows(args.development)
    if len(public) != 72 or len(development) != 80 or any(row["split"] != "development" for row in development):
        raise ValueError("Unexpected cohort")
    torch.backends.cuda.matmul.allow_tf32 = False
    parser = ComparisonParser(args.model)
    for reader in parser.readers.values():
        reader.broadcast = True
    for mode in ARMS:
        text = public[0]["text"]
        parser_run(text, spans(text), parser.readers, mode)
    panels = {}
    for name, cohort in (("development", development), ("controls", public)):
        panels[name] = request_panel(cohort, parser.readers, args.reverse, name == "controls")
        print(name, panels[name]["counts"], flush=True)
    panels["forced_long_target_blocks"] = forced_blocks(public, parser.readers)
    write(args.output, dict(
        panels=panels, model=parser.readers["bf16"].provenance["revision"], gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse, script_sha256=digest(Path(__file__)),
        inputs={key: digest(getattr(args, key)) for key in ("development", "controls")},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "model", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
