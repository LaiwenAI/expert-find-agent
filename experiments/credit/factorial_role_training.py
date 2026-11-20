"""Factor synthetic role examples and optimizer on the same frozen benchmark."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import calibrated_role_contrasts as study
from experiments.credit import role_swap_training as swaps


DATASETS = ("single_clause", "other_role_swap", "near_role_swap")
OPTIMIZERS = ("ridge", "cross_entropy")
TRAINING = tuple(f"{data}_{opt}" for data in DATASETS for opt in OPTIMIZERS)
STEPS = 120


def fit(encoder, training_path):
    import torch
    from torch.nn.functional import cross_entropy
    from experiments.credit.train_role_decision import encode, ridge

    seeds = [dict(role=role, text=text) for role, texts in swaps.descriptions(training_path).items()
             for text in texts]
    generated = swaps.generate(training_path)["datasets"]
    if set(generated) != set(DATASETS) or len({len(x) for x in generated.values()}) != 1:
        raise ValueError("Unmatched generated data")
    with torch.inference_mode():
        base = encode(encoder, [row["text"] for row in seeds])
    weights, metadata = {}, {}
    for dataset in DATASETS:
        examples = generated[dataset]
        started = perf_counter()
        y = torch.tensor([study.ROLE_ID[row["role"]] for row in seeds + examples], device="cuda")
        with torch.inference_mode():
            batches = (
                encode(encoder, [row["text"] for row in examples[i:i + 32]])
                for i in range(0, len(examples), 32)
            )
            features = torch.cat((base, *batches))
            initial = ridge(features, y)
            torch.cuda.synchronize()
        ridge_seconds = perf_counter() - started
        weights[f"{dataset}_ridge"] = initial
        began = perf_counter()
        x = features.clone()
        parameter = torch.nn.Parameter(initial.clone() * 10)
        optimizer = torch.optim.AdamW([parameter], lr=0.002)
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(x @ parameter.T, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        weights[f"{dataset}_cross_entropy"] = parameter.detach()
        metadata[dataset] = dict(
            seeds=len(seeds), generated=len(examples), parameters=parameter.numel(),
            ridge_and_encoding_seconds=ridge_seconds, cross_entropy_seconds=perf_counter() - began, steps=STEPS,
        )
    return weights, metadata


def summarize(measured, thresholds, cases, clusters):
    references = {"reader": "reader"}
    references.update({f"{data}_{opt}": f"single_clause_{opt}" for data in DATASETS for opt in OPTIMIZERS})
    scores = study.report(measured, thresholds, cases, clusters, references=references)
    groups = [clusters[row["paper"]] for row in cases]
    predicted = {arm: np.asarray([values for values, _ in rows]) > thresholds[arm]
                 for arm, rows in measured.items() if arm in TRAINING}
    flags = {arm: dict(positive=out[:, 0].astype(int), rival=out[:, 1].astype(int),
                       clean=(out[:, 0] & ~out[:, 1]).astype(int)) for arm, out in predicted.items()}

    def metrics(plus, minus):
        return {key: clustered_paired_interval(sum(flags[arm][key] for arm in plus)
                                               - sum(flags[arm][key] for arm in minus), groups)
                for key in ("clean", "positive", "rival")}

    contrasts = {}
    for opt in OPTIMIZERS:
        for control in DATASETS[:2]:
            contrasts[f"near_role_swap_minus_{control}_{opt}"] = metrics(
                [f"near_role_swap_{opt}"], [f"{control}_{opt}"],
            )
    for data in DATASETS:
        contrasts[f"{data}_cross_entropy_minus_ridge"] = metrics(
            [f"{data}_cross_entropy"], [f"{data}_ridge"])
    for data in DATASETS[1:]:
        contrasts[f"{data}_optimizer_interaction"] = metrics(
            [f"{data}_cross_entropy", "single_clause_ridge"],
            [f"{data}_ridge", "single_clause_cross_entropy"])
    return dict(arms=scores, contrasts=contrasts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "legacy_transfer", "dev_xml", "test_xml", "data", "examples", "embedding", "reranker", "output"):
        parser.add_argument(arg, type=Path)
    study.run(**vars(parser.parse_args()), fit_heads=fit, training=TRAINING,
              summarize=summarize, script_path=Path(__file__))
