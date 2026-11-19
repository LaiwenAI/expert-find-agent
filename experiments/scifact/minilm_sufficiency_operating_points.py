"""Compare evidence retention at common omission-risk targets."""

import argparse
from itertools import permutations
import json
import os
from pathlib import Path

import numpy as np

from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.minilm_sufficient_facts import (
    INPUTS, INSUFFICIENT, SEEDS, bootstrap_delta, load_pairs,
)
from experiments.scifact.minilm_transfer import apply_weights


CAP_PERCENT = (5, 10, 20)


def operating_threshold(negative_margins, cap_percent):
    """Lowest observed threshold that meets the empirical negative cap."""
    allowed = cap_percent * len(negative_margins) // 100
    if not 0 <= allowed < len(negative_margins) or not np.isfinite(negative_margins).all():
        raise ValueError("Need finite negative calibration scores")
    rank = len(negative_margins) - allowed - 1
    return float(np.partition(negative_margins, rank)[rank])


def panel_metrics(panel, scores, threshold, reference=None):
    accepted = scores > threshold
    essential = panel["essential"]
    selectors = {
        "essential_pruned": (essential, accepted[1]),
        "essential_pair_correct": (essential, accepted[0] & ~accepted[1]),
        "full_support": (np.ones_like(essential), accepted[0]),
        "sufficient_pruned": (~essential, accepted[1]),
        "sufficient_pair_correct": (~essential, accepted[0] & accepted[1]),
    }
    metrics = {}
    decisions = {}
    for metric, (mask, predicted) in selectors.items():
        decisions[metric] = predicted[mask]
        metrics[metric] = {"count": int(predicted[mask].sum()), "n": int(mask.sum())}
        if reference is not None:
            delta = decisions[metric].astype(float) - reference[metric]
            metrics[metric]["paired_delta"] = float(delta.mean())
            metrics[metric]["cluster_ci95"] = bootstrap_delta(delta, panel["claim"][mask])
    return metrics, decisions


def panel_subset(panel, selector):
    return {key: value[selector] for key, value in panel.items()}


def assess(panels, scores):
    report = {}
    for calibration_name, target_name in permutations(panels, 2):
        calibration, target = panels[calibration_name], panels[target_name]
        overlap = np.isin(target["claim"], calibration["claim"])
        if overlap.all():
            raise ValueError("No held-out target claims")
        target = panel_subset(target, ~overlap)
        report[f"{calibration_name}_to_{target_name}"] = {
            "excluded_shared_claim_rows": int(overlap.sum()), "caps": {},
        }
        settings = report[f"{calibration_name}_to_{target_name}"]["caps"]
        for cap in CAP_PERCENT:
            settings[str(cap)] = arms = {}
            transfer_base, envelope_base = None, None
            for arm, arm_scores in scores.items():
                cal_margins = arm_scores[calibration_name]
                target_margins = arm_scores[target_name][:, ~overlap]
                threshold = operating_threshold(cal_margins[1, calibration["essential"]], cap)
                estimated, _ = panel_metrics(calibration, cal_margins, threshold)
                observed, decisions = panel_metrics(target, target_margins, threshold, transfer_base)
                # This descriptive envelope uses target labels to match the target risk.
                matched = operating_threshold(target_margins[1, target["essential"]], cap)
                envelope, envelope_decisions = panel_metrics(target, target_margins, matched, envelope_base)
                if arm == "frozen":
                    transfer_base, envelope_base = decisions, envelope_decisions
                arms[arm] = dict(threshold=threshold, calibration=estimated, cross_dataset=observed,
                                 envelope_threshold=matched, retrospective_at_cap=envelope)
    return report


def run(dataset_dir, scifact_report, scifact_weights, vitaminc_report, vitaminc_weights,
        checkpoint, prior_results, output):
    import torch
    from experiments.scifact.small_nli_verifier import load, measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output path and Slurm CUDA")
    previous = json.loads(prior_results.read_text())
    if (previous["inputs"] != INPUTS or previous["scifact_report_sha256"] != digest(scifact_report)
            or previous["vitaminc_report_sha256"] != digest(vitaminc_report)):
        raise ValueError("Expected the same data and training cohorts")
    pairs = {name: load_pairs(dataset_dir / f"{name}.jsonl", expected) for name, expected in INPUTS.items()}
    panels = {
        name: dict(
            claim=np.asarray([" ".join(row["claim"].casefold().split()) for row in rows]),
            essential=np.asarray([row["label"] == INSUFFICIENT for row in rows]),
        )
        for name, rows in pairs.items()
    }
    weights = {"frozen": None}
    weights.update({f"rationale_pair_{seed}": scifact_weights for seed in SEEDS})
    weights.update({f"risk_tail_{seed}": vitaminc_weights for seed in SEEDS})
    scores = {}
    for arm, directory in weights.items():
        tokenizer, model, positions, provenance = load(checkpoint, "minilm")
        if previous["model_revision"] != provenance["revision"]:
            raise ValueError("Changed verifier revision")
        original = previous["report"][arm]
        if directory is not None:
            apply_weights(model, directory / f"{arm}.safetensors", original["weight_sha256"])
        scores[arm] = {}
        for name, rows in pairs.items():
            margins = []
            for view in ("original", "pruned"):
                batch = [dict(source=row[view], criterion=row["claim"]) for row in rows]
                estimates, _ = measure(tokenizer, model, positions, batch, batch_size=1)
                if any(item is None for item in estimates):
                    raise ValueError("All replay views must still fit the checkpoint")
                margins.append(np.asarray([item[1] for item in estimates]))
            values = np.asarray(margins)
            replay, _ = panel_metrics(panels[name], values, original["threshold"])
            prior = original["datasets"][name]["transfer"]["groups"]
            for metric, group, key in (
                ("essential_pruned", "essential", "pruned_accepted"),
                ("essential_pair_correct", "essential", "both_correct"),
                ("sufficient_pruned", "sufficient", "pruned_accepted"),
                ("sufficient_pair_correct", "sufficient", "both_correct"),
            ):
                if replay[metric]["count"] != prior[group][key]:
                    raise ValueError(f"Prior replay count changed: {arm} {name} {metric}")
            scores[arm][name] = values
        del model
        print(arm, "replayed", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            priors_sha256=digest(prior_results), script_sha256=digest(Path(__file__)),
                            replay_runner_sha256=digest(Path(load_pairs.__code__.co_filename)),
                            inputs=INPUTS, cap_percent=CAP_PERCENT, report=assess(panels, scores)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_dir", "scifact_report", "scifact_weights", "vitaminc_report", "vitaminc_weights",
                 "checkpoint", "prior_results", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
