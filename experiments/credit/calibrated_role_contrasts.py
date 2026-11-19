"""Calibrate equal-data role decision heads on source- and wording-disjoint cases."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import factored_article_role_verification as factored
from experiments.credit import frozen_scope_role_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit import role_swap_training as swaps
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json

TRAINING = ("single_clause", "other_role_swap", "near_role_swap", "near_cross_entropy", "near_rank_loss")
ARMS = ("reader", *TRAINING)
ROLE_ID = {role: i for i, role in enumerate(ROLES)}


def fit(encoder, train_path):
    import torch
    from torch.nn.functional import cross_entropy, softplus
    from experiments.credit.train_role_decision import encode, ridge

    seeds = [dict(role=role, text=text) for role, texts in swaps.descriptions(train_path).items() for text in texts]
    generated = swaps.generate(train_path)["datasets"]
    weights, meta = {}, {}
    with torch.inference_mode():
        base = encode(encoder, [r["text"] for r in seeds])
        for mode, example in generated.items():
            rows = seeds + example
            y = torch.tensor([ROLE_ID[row["role"]] for row in rows], device="cuda")
            tick = perf_counter()
            vectors = torch.cat((base, *[encode(encoder, [r["text"] for r in example[i:i + 32]])
                                        for i in range(0, len(example), 32)]))
            fitted = ridge(vectors, y)
            weights[mode] = fitted
            torch.cuda.synchronize()
            meta[mode] = dict(examples=len(rows), seconds=perf_counter() - tick)
            if mode == "near_role_swap":
                near_vectors, near_labels = vectors, y
    # Reuse the exact near-swap rows. Only the optimization objective changes.
    near = generated["near_role_swap"]
    x = near_vectors.clone()
    y = near_labels.clone()
    initial = weights["near_role_swap"].clone()
    alternative = torch.tensor([ROLE_ID[near[i ^ 1]["role"]] for i in range(len(near))], device="cuda")
    if any(near[i]["group"] != near[i ^ 1]["group"] or near[i]["role"] == near[i ^ 1]["role"]
           for i in range(len(near))):
        raise ValueError("Expected matched credit swaps")
    steps = 120
    targets = y[-len(near):]
    for mode in ("near_cross_entropy", "near_rank_loss"):
        start = perf_counter()
        parameter = torch.nn.Parameter(initial * 10)
        optimizer = torch.optim.AdamW([parameter], lr=.002)
        for _ in range(steps):
            scores = x @ parameter.T
            loss = cross_entropy(scores, y)
            if mode == "near_rank_loss":
                matched = scores[-len(near):]
                contrast = matched.gather(1, targets[:, None]) - matched.gather(1, alternative[:, None])
                loss = loss + softplus(-contrast).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        weights[mode] = parameter.detach()
        meta[mode] = dict(examples=len(y), steps=steps, optimized_seconds=perf_counter() - start,
                          parameters=parameter.numel())
    return weights, meta


def query_bank(path):
    grouped = defaultdict(list)
    for row in json.loads(path.read_bytes())["examples"]:
        grouped[row["role"]].append(row["text"])
    if set(grouped) != set(ROLES):
        raise ValueError("Calibration bank must cover all supported roles")
    return dict(grouped)


def normalized_names(cases):
    return {" ".join(legacy.NAME_PARTS.findall(name)).casefold() for case in cases for name in case["names"]}


def observe(cases, bank, encoder, weights, reader):
    import torch
    from experiments.credit.train_role_decision import encode

    measured = {mode: [] for mode in ARMS}
    stacked = torch.stack([weights[mode] for mode in TRAINING])
    for i, item in enumerate(cases):
        role, foe = item["payloads"][0][1], item["payloads"][1][1]
        phrase = min(bank[role], key=lambda text: legacy.stable(item["doi"], role, text))
        mode_inputs = [(phrase, candidate) for candidate in (role, foe)]
        for name in ("reader", "heads")[i % 2:] + ("reader", "heads")[:i % 2]:
            started = perf_counter()
            if name == "reader":
                scores, clipped = reader.score_pairs(mode_inputs, task=factored.ROLE_TASK)
                if clipped or not np.isfinite(scores).all():
                    raise ValueError("Invalid reader result")
                measured[name].append((scores, (perf_counter() - started) * 1000))
            else:
                with torch.inference_mode():
                    vector = encode(encoder, [phrase])[0]
                    # One query embedding and a shared forward across all experimental heads.
                    scores = (stacked @ vector)[:, [ROLE_ID[role], ROLE_ID[foe]]].cpu().tolist()
                elapsed = (perf_counter() - started) * 1000
                for arm, values in zip(TRAINING, scores, strict=True):
                    measured[arm].append((values, elapsed))
    return measured


def calibrate(scores):
    rows = np.asarray([row[0] for row in scores])
    unique = np.unique(rows)
    cuts = np.concatenate(([np.nextafter(unique[0], -np.inf)], unique))
    options = []
    for threshold in cuts:
        positive, foil = (rows > threshold).T
        if foil.sum() <= len(rows) // 10:
            options.append((int((positive & ~foil).sum()), -int(foil.sum()), float(threshold)))
    return max(options)[-1]


def report(measured, thresholds, cases, clusters):
    labels, result = {}, {}
    for arm, samples in measured.items():
        scores = np.asarray([row[0] for row in samples])
        accepted, rival = (scores > thresholds[arm]).T
        labels[arm] = accepted & ~rival
        result[arm] = dict(
            cases=len(scores), positive=int(accepted.sum()), wrong_role=int(rival.sum()),
            clean=int(labels[arm].sum()), ordered=int((scores[:, 0] > scores[:, 1]).sum()),
            p95_ms=float(np.quantile([row[1] for row in samples], .95)),
        )
    groups = [clusters[case["paper"]] for case in cases]
    for arm in ARMS:
        ref = "near_cross_entropy" if arm == "near_rank_loss" else ("single_clause" if arm in TRAINING else "reader")
        result[arm]["clean_difference"] = clustered_paired_interval(
            labels[arm].astype(int) - labels[ref].astype(int), groups,
        )
    return result


def run(old, legacy_transfer, dev_xml, test_xml, data, examples, embedding, reranker, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.role_prototypes import normalized
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ or output.exists():
        raise ValueError("Require Slurm CUDA and a new output")
    dev_manifest = data / "plos-credit-transfer-v1/manifest.json"
    test_manifest = data / "plos-credit-external-v1/manifest.json"
    train_path = data / "plos-credit-edits-v1/role-training.json"
    test_banks = {"straightforward": data / "plos-credit-transfer-v1/phrases.json",
                  "boundary": data / "plos-credit-external-v1/boundary-phrases.json"}
    calibration, dev_groups, dev_excluded = transfer.make_cases(old, legacy_transfer, dev_xml, dev_manifest)
    test, test_groups, test_excluded = transfer.make_cases(old, legacy_transfer, test_xml, test_manifest)
    forbidden_names = normalized_names(test)
    forbidden_dois = {case["doi"] for case in test}
    before = len(calibration)
    calibration = [case for case in calibration if case["doi"] not in forbidden_dois
                   and forbidden_names.isdisjoint(normalized_names([case]))]
    banks = {view: json.loads(path.read_bytes()) for view, path in test_banks.items()}
    calibration_bank = query_bank(examples)
    seen = {normalized(text) for rows in swaps.descriptions(train_path).values() for text in rows}
    for bank in (calibration_bank, *banks.values()):
        current = {normalized(text) for rows in bank.values() for text in rows}
        if seen & current:
            raise ValueError("Description wording overlaps across an evaluation split")
        seen.update(current)
    if not calibration or not test:
        raise ValueError("Missing source-disjoint test cases")
    embed_info = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    reader = Reranker(reranker)
    weights, fit_summary = fit(encoder, train_path)
    observe(calibration[:4], calibration_bank, encoder, weights, reader)
    measured_dev = observe(calibration, calibration_bank, encoder, weights, reader)
    thresholds = {arm: calibrate(rows) for arm, rows in measured_dev.items()}
    results = {view: report(observe(test, bank, encoder, weights, reader), thresholds, test, test_groups)
               for view, bank in banks.items()}
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), developer_source_sha256=digest(dev_manifest),
        test_source_sha256=digest(test_manifest), training_sha256=digest(train_path), calibration_bank_sha256=digest(examples),
        wording_sha256={view: digest(path) for view, path in test_banks.items()},
        checkpoints=dict(embedding=embed_info["revision"], reader=reader.provenance["revision"]),
        thresholds=thresholds, training=fit_summary, dev_excluded=dev_excluded, test_excluded=test_excluded,
        dev_excluded_for_test_overlap=before - len(calibration),
        calibration_papers=len({case["doi"] for case in calibration}), test_papers=len(test_groups),
        calibration=report(measured_dev, thresholds, calibration, dev_groups), panels=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "legacy_transfer", "dev_xml", "test_xml", "data", "examples", "embedding", "reranker", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
