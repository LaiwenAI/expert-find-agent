"""Test whether frozen evidence gates respond to human-judged information removal."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

import numpy as np

from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.minilm_transfer import apply_weights


INPUTS = {
    "fever": "c6b8910a3d4a358140ed0bd4fdbb5f94acf957c37b80b4732a1c0577e5979532",
    "hover": "60ccd362b68fbb30f74f91c6a2240f0a35fe7fb9f48d59e914e15f33488d0950",
}
REMOVAL = re.compile(r'<span style="color:red;">(.*?)</span>', re.DOTALL)
SUFFICIENT = {"ENOUGH -- IRRELEVANT", "ENOUGH -- REPEATED"}
INSUFFICIENT = "NOT ENOUGH"
SEEDS = (20261119, 20261120)


def normalize_source(source):
    return " ".join(re.sub(r"\]\](?=\S)", "]] ", source).split())


def load_pairs(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed SufficientFacts input: {path.name}")
    result = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["label_before"] == "REFUTES":
            continue
        if row["label_before"] not in {"SUPPORTED", "SUPPORTS"}:
            raise ValueError("Unexpected original label")
        label = row["label_after"]
        if label not in SUFFICIENT | {INSUFFICIENT}:
            raise ValueError("Unexpected human sufficiency label")
        original, found = REMOVAL.subn(r"\1", row["text_orig"])
        pruned = normalize_source(REMOVAL.sub("", row["text_orig"]))
        original = normalize_source(original)
        released = normalize_source(" ".join(f"[[{title}]] {text}" for title, text in row["evidence"]))
        if pruned != released:
            raise ValueError("Removed view must match the released evidence")
        if not found or "<span" in pruned or "</span>" in pruned or original == pruned:
            raise ValueError("Missing or malformed removal markup")
        result.append(dict(claim=row["claim"], original=original, pruned=pruned, label=label))
    return result


def bootstrap_delta(deltas, claims):
    _, inverse = np.unique(claims, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=deltas)
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(counts), size=(4000, len(counts)))
    averages = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(x) for x in np.quantile(averages, [0.025, 0.975])]


def drops_auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    negative = np.sort(negative)
    below = np.searchsorted(negative, positive, side="left")
    at_or_below = np.searchsorted(negative, positive, side="right")
    return float(np.mean((below + at_or_below) / (2 * len(negative))))


def summarize(pairs, views, threshold, reference):
    original, pruned = (views[view][0] for view in ("original", "pruned"))
    claims = np.asarray([" ".join(row["claim"].casefold().split()) for row in pairs])
    needs = np.asarray([row["label"] == INSUFFICIENT for row in pairs])
    eligible = np.isfinite(original) & np.isfinite(pruned)
    accepted_original = original > threshold
    accepted_pruned = pruned > threshold
    correct = accepted_original & (accepted_pruned == ~needs) & eligible
    drop = np.full(original.shape, np.nan)
    np.subtract(original, pruned, out=drop, where=eligible)
    result = {"pairs": len(pairs), "claims": len(set(claims)), "labels": dict(Counter(r["label"] for r in pairs)),
              "over_limit": {"original": int(np.sum(~np.isfinite(original))),
                             "pruned": int(np.sum(~np.isfinite(pruned)))},
              "margin_drop_auc": drops_auc(drop[needs & eligible], drop[~needs & eligible]),
              "groups": {}}
    for name, mask in (("essential", needs), ("sufficient", ~needs)):
        kept = mask & eligible
        result["groups"][name] = dict(
            pairs=int(mask.sum()), both_fit=int(kept.sum()),
            original_accepted=int(np.sum(mask & accepted_original)),
            pruned_accepted=int(np.sum(mask & accepted_pruned)),
            pruned_accepted_given_original=int(np.sum(mask & accepted_pruned & accepted_original)),
            both_correct=int(np.sum(mask & correct)),
            median_margin_drop=float(np.median(drop[kept])) if kept.any() else None,
        )
        if reference is not None:
            intersection = kept & reference["eligible"]
            delta = correct[intersection].astype(float) - reference["correct"][intersection]
            result["groups"][name]["paired_delta_vs_frozen"] = dict(
                pairs=int(intersection.sum()), estimate=float(delta.mean()),
                cluster_ci95=bootstrap_delta(delta, claims[intersection]),
            )
    for view, (margins, times) in views.items():
        result[f"{view}_p95_ms"] = float(np.quantile(times[np.isfinite(margins)], 0.95))
    return result, dict(eligible=eligible, correct=correct)


def run(dataset_dir, scifact_report, scifact_weights, vitaminc_report, vitaminc_weights, checkpoint, output):
    import torch
    from experiments.scifact.small_nli_verifier import load, measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output path and Slurm CUDA")
    pairs = {name: load_pairs(dataset_dir / f"{name}.jsonl", expected) for name, expected in INPUTS.items()}
    sci, vit = [json.loads(path.read_text()) for path in (scifact_report, vitaminc_report)]
    arms = {"frozen": (None, None, sci["report"]["frozen"]["threshold"])}
    for seed in SEEDS:
        name = f"rationale_pair_{seed}"
        entry = sci["report"][name]
        arms[name] = (scifact_weights / f"{name}.safetensors",
                      entry["training"]["weight_sha256"], entry["threshold"])
        name = f"risk_tail_{seed}"
        entry = vit["report"][name]
        arms[name] = (vitaminc_weights / f"{name}.safetensors",
                      entry["training"]["weight_sha256"], entry["operating_points"]["strict"]["threshold"])

    reports, frozen = {}, {}
    for arm, (weights, expected, threshold) in arms.items():
        tokenizer, model, positions, provenance = load(checkpoint, "minilm")
        if not sci["model_revision"] == vit["model_revision"] == provenance["revision"]:
            raise ValueError("Changed verifier revision")
        if weights:
            apply_weights(model, weights, expected)
        measure(tokenizer, model, positions, [{"source": "A test source.", "criterion": "A test claim."}],
                batch_size=1)
        reports[arm] = dict(threshold=threshold, weight_sha256=expected, datasets={})
        for name, rows in pairs.items():
            views = {}
            for view in ("original", "pruned"):
                inputs = [dict(source=row[view], criterion=row["claim"]) for row in rows]
                scores, times = measure(tokenizer, model, positions, inputs, batch_size=1)
                views[view] = np.asarray([score[1] if score is not None else -np.inf for score in scores]), np.asarray(times)
            summary, decisions = summarize(rows, views, threshold, frozen.get((name, "transfer")))
            raw, raw_decisions = summarize(rows, views, 0.0, frozen.get((name, "raw")))
            reports[arm]["datasets"][name] = dict(transfer=summary, positive_margin=raw)
            if arm == "frozen":
                frozen[name, "transfer"] = decisions
                frozen[name, "raw"] = raw_decisions
        del model
        print(arm, "complete", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                           inputs=INPUTS, script_sha256=digest(Path(__file__)), model_revision=provenance["revision"],
                           scifact_report_sha256=digest(scifact_report), vitaminc_report_sha256=digest(vitaminc_report),
                           report=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset_dir", "scifact_report", "scifact_weights", "vitaminc_report", "vitaminc_weights",
                "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
