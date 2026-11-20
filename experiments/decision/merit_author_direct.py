"""Isolate long generation and profile depth at fixed MERIT assessor weights."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from expert_search.merit_suitability import build_case, prompts
from expert_search.statistics import clustered_paired_interval
from experiments.decision import merit_author_assessor as author
from experiments.decision.merit_compact_source_head import cutoffs
from experiments.decision.merit_evidence_dispatch import read_cases, select
from experiments.decision.merit_lightweight_decision import split_cases
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("original_full_prefill", "direct_full", "direct_two")
SUFFIX = "[FINAL_LABEL]\n"


def label_tokens(tokenizer):
    prefix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    candidates = [tokenizer.encode(SUFFIX + digit, add_special_tokens=False) for digit in ("0", "1")]
    if any(ids[:-1] != prefix for ids in candidates) or candidates[0][-1] == candidates[1][-1]:
        raise ValueError("The label continuation is not a single distinct token")
    return [ids[-1] for ids in candidates]


def compact_row(row, case):
    remaining = list(select(case, complementary=True)[0])
    selected = []
    for paper, text in zip(row["author"]["papers"], case.papers, strict=True):
        if text in remaining:
            selected.append(paper)
            remaining.remove(text)
    if remaining or len(selected) != 2:
        raise ValueError("Could not map selected evidence to two original sources")
    return {**row, "author": {**row["author"], "papers": selected}}


def direct_prompt(tokenizer, official, row, index, case, arm):
    instructions, separator, _ = official["action_system_prompt"].partition("Instruction:")
    if not separator:
        raise ValueError("Unexpected original instructions")
    system = (official["action_system_prompt"] if arm == "original_full_prefill" else
              instructions + "Return the binary suitability label under [FINAL_LABEL].")
    if arm == "direct_two":
        row = compact_row(row, case)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": build_case(row, index, official)["input"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + SUFFIX


def score_cases(model, tokenizer, label_ids, official, records, batch_size, *, time_prompt_assembly=True):
    scores = {arm: [] for arm in ARMS}
    seconds = {arm: [] for arm in ARMS}
    lengths = {arm: [] for arm in ARMS}
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        # Rotate order without sharing prepared prompts or tokenized inputs.
        rotate = (offset // batch_size) % len(ARMS)
        for arm in ARMS[rotate:] + ARMS[:rotate]:
            start = perf_counter()
            texts = [direct_prompt(tokenizer, official, row, index, case, arm) for index, row, case in batch]
            if not time_prompt_assembly:
                start = perf_counter()
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda:0")
            if inputs.input_ids.shape[1] > author.MAX_PROMPT:
                raise ValueError("Prefill exceeds the common input budget")
            position_ids = inputs.attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(inputs.attention_mask == 0, 0)
            logits = model(
                **inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1,
            ).logits[:, -1, label_ids].float()
            margins = (logits[:, 1] - logits[:, 0]).tolist()
            seconds[arm].append(perf_counter() - start)
            scores[arm].extend(margins)
            lengths[arm].extend(inputs.attention_mask.sum(dim=1).tolist())
    return scores, seconds, lengths


@torch.inference_mode()
def run(args):
    if digest(args.prompts) != author.PROMPT_SHA:
        raise ValueError("Unexpected official prompt")
    for name, expected in author.WEIGHTS.items():
        if digest(args.checkpoint / name) != expected:
            raise ValueError(f"Unexpected checkpoint hash: {name}")
    cases = read_cases(args.data)
    rows = json.loads(args.data.read_bytes())
    splits = split_cases(cases)
    counts = {name: len(splits[name]) for name in ("calibration", "test")}
    if counts != {"calibration": 125, "test": 129}:
        raise ValueError("Changed shared evaluation split")
    indices = {id(case): i for i, case in enumerate(cases)}
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, padding_side="left", local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = label_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
    ).to("cuda:0").eval()
    official = prompts(args.prompts)
    records = {}
    for name in ("calibration", "test"):
        subset = [(indices[id(case)], case) for _, case in splits[name]]
        records[name] = [(index, rows[index], case) for index, case in subset]
    calibration_labels = [case.label for _, case in splits["calibration"]]
    expected = np.asarray([case.label for _, case in splits["test"]])
    groups = [group for group, _ in splits["test"]]
    calibration_scores, _, _ = score_cases(model, tokenizer, label_ids, official, records["calibration"], 1)
    test_scores, test_seconds, test_lengths = score_cases(model, tokenizer, label_ids, official, records["test"], 1)
    print(f"Completed serial scoring for {len(expected)} held records.", flush=True)
    batch_scores, batch_seconds, _ = score_cases(
        model, tokenizer, label_ids, official, records["test"], args.matched_batch_size,
        time_prompt_assembly=False,
    )
    print(f"Completed scoring in batches of {args.matched_batch_size}.", flush=True)
    conditions = {
        arm: {"unadjusted": 0.0, **cutoffs(calibration_scores[arm], calibration_labels)}
        for arm in ARMS
    }
    report = {}
    for arm in ARMS:
        values = np.asarray(test_scores[arm])
        matched = np.asarray(batch_scores[arm])
        results = {}
        for condition, cutoff in conditions[arm].items():
            prediction = (values >= cutoff).astype(int)
            batch_prediction = (matched >= cutoff).astype(int)
            baseline = np.asarray(test_scores["direct_full"]) >= conditions["direct_full"][condition]
            results[condition] = {
                **author.metrics(list(zip(expected, prediction, strict=True))),
                "matched_batch": {
                    **author.metrics(list(zip(expected, batch_prediction, strict=True))),
                    "different_from_serial": int(np.count_nonzero(batch_prediction != prediction)),
                },
                "different_from_direct_full": int(np.count_nonzero(prediction != baseline)),
                "accuracy_delta_vs_direct_full": clustered_paired_interval(
                    (prediction == expected).astype(int) - (baseline == expected).astype(int), groups,
                ),
            }
        report[arm] = {
            "thresholds": conditions[arm],
            "test": results,
            "serial_p95_ms": float(np.percentile(test_seconds[arm], 95) * 1000),
            "median_prompt_tokens": float(np.median(test_lengths[arm])),
            "matched_batch": {
                "size": args.matched_batch_size,
                "batches": len(batch_seconds[arm]),
                "total_seconds": sum(batch_seconds[arm]),
                "p95_batch_seconds": float(np.percentile(batch_seconds[arm], 95)),
            },
        }
    return {
        "dataset_sha256": author.DATA_SHA, "split": author.SALT, "counts": counts,
        "checkpoint": author.CHECKPOINT, "revision": author.REVISION,
        "script_sha256": digest(Path(__file__)), "gpu": torch.cuda.get_device_name(), "routes": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "prompts", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--matched-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.matched_batch_size < 1:
        parser.error("--matched-batch-size must be positive")
    if args.output.exists():
        parser.error("output already exists")
    write_json(args.output, run(args))


if __name__ == "__main__":
    main()
