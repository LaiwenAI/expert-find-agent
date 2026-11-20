"""Evaluate the released MERIT assessor on the shared profile-disjoint holdout."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from expert_search.merit_suitability import build_case, prompts
from experiments.decision.merit_evidence_dispatch import DATA_SHA, read_cases
from experiments.decision.merit_lightweight_decision import SALT, split_cases
from experiments.followir.core17_reranking import digest, write_json

CHECKPOINT = "Luli3220/MERIT-4B-reviewer-assessor"
REVISION = "75daec26184881f85fbc76f3ffb67bd218b74d26"
RELEASE = "5fad0ea06148c3c8817aac841de46c98a75eb40c"
PROMPT_SHA = "03376da1483a24b2f193d9c69528c04563fddb343ed9b62662451f9b8eed86ec"
WEIGHTS = {
    "model-00001-of-00002.safetensors": "65f8f4b6f84254c65b659c6a01e9ce793d7a4e85b68f47e45a9afd53108d081a",
    "model-00002-of-00002.safetensors": "c1bf03b1d2beb14ed7d5729ff082fedc3d6be7ebfec6307816801e6568437a85",
}
MAX_PROMPT = 5120
MAX_GENERATION = 4096


def author_label(text):
    """Retain the published parser's precedence and expose its fallback rate."""
    thinking = r"<think>.*?(?:</think>|<\[PLHD21_never_used_51bce0c785ca2f68081bfa7d91973934\]>)"
    clean = re.sub(thinking, "", text, flags=re.DOTALL).strip()
    if "[FINAL_LABEL]" in clean:
        match = re.search(r"\b(0|1)\b", clean.rsplit("[FINAL_LABEL]", 1)[-1])
        if match:
            return int(match.group(1)), "final_field"
    match = re.search(r"<\s*(0|1)\s*>", clean)
    if match:
        return int(match.group(1)), "angle_fallback"
    return 0, "default_negative"


def metrics(pairs):
    counts = Counter(pairs)
    tp, tn, fp, fn = (counts[pair] for pair in ((1, 1), (0, 0), (0, 1), (1, 0)))
    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": (tp + tn) / len(pairs) if pairs else None,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) and (tn + fp) else None,
        "positive_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def holdout(data):
    cases = read_cases(data)
    indices = {id(case): i for i, case in enumerate(cases)}
    test = split_cases(cases)["test"]
    if (len(test), sum(case.label for _, case in test), len({group for group, _ in test})) != (129, 45, 60):
        raise ValueError("The shared MERIT holdout has changed")
    rows = json.loads(data.read_bytes())
    return [(indices[id(case)], rows[indices[id(case)]], case.label) for _, case in test]


def prepare(tokenizer, records, official):
    eligible = []
    skipped = Counter()
    for index, row, label in records:
        messages = [
            {"role": "system", "content": official["action_system_prompt"]},
            {"role": "user", "content": build_case(row, index, official)["input"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        length = len(tokenizer.encode(text))
        if length > MAX_PROMPT:
            skipped[label] += 1
        else:
            eligible.append((text, label))
    return eligible, skipped


@torch.inference_mode()
def run(args):
    if digest(args.prompts) != PROMPT_SHA:
        raise ValueError("Unexpected upstream prompt revision")
    records = holdout(args.data)
    for filename, expected in WEIGHTS.items():
        if digest(args.checkpoint / filename) != expected:
            raise ValueError(f"Unexpected checkpoint hash: {filename}")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, padding_side="left", local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eligible, skipped = prepare(tokenizer, records, prompts(args.prompts))
    if not eligible:
        raise ValueError("No prompts within the official length cap")
    print(f"Eligible: {len(eligible)} / {len(records)}. Skipped labels: {dict(skipped)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False,
    ).to("cuda:0").eval()
    eos = model.generation_config.eos_token_id
    eos_tokens = {eos} if isinstance(eos, int) else set(eos or [])
    pairs, batch_seconds = [], []
    parse_counts = Counter()
    cap_without_eos = 0
    for offset in range(0, len(eligible), args.batch_size):
        batch = eligible[offset:offset + args.batch_size]
        start = perf_counter()
        inputs = tokenizer(
            [text for text, _ in batch], return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_PROMPT,
        ).to("cuda:0")
        output = model.generate(
            **inputs, max_new_tokens=MAX_GENERATION, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
        continuations = output[:, inputs.input_ids.shape[1]:].cpu()
        decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        batch_seconds.append(perf_counter() - start)
        for generated, text, (_, label) in zip(continuations.tolist(), decoded, batch, strict=True):
            prediction, route = author_label(text)
            pairs.append((label, prediction))
            parse_counts[route] += 1
            if len(generated) == MAX_GENERATION and eos_tokens and eos_tokens.isdisjoint(generated):
                cap_without_eos += 1
        print(f"Generated {len(pairs)} / {len(eligible)} in {batch_seconds[-1]:.2f} s this batch.", flush=True)
    return {
        "dataset": {"release": RELEASE, "sha256": DATA_SHA, "split": SALT, "held": len(records)},
        "checkpoint": {"repository": CHECKPOINT, "revision": REVISION, "weight_sha256": WEIGHTS},
        "configuration": {
            "batch_size": args.batch_size, "prompt_sha256": PROMPT_SHA, "max_prompt": MAX_PROMPT,
            "max_generation": MAX_GENERATION, "seed": 42, "dtype": "bfloat16",
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "evaluated": len(pairs), "skipped_positive": skipped[1], "skipped_negative": skipped[0],
        "label_parse_paths": dict(parse_counts), "generation_cap_without_eos": cap_without_eos,
        "author_protocol": metrics(pairs),
        "all_held_with_skips_as_negative": metrics(pairs + [(1, 0)] * skipped[1] + [(0, 0)] * skipped[0]),
        "batch_timing": {
            "total_seconds": sum(batch_seconds), "batches": len(batch_seconds),
            "p95_batch_seconds": float(np.percentile(batch_seconds, 95)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error("output already exists")
    write_json(args.output, run(args))


if __name__ == "__main__":
    main()
