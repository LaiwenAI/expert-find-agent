"""Fit a depth router on QASPER development papers and time its frozen test policy."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.qasper import certified_reranking as base


ACTIONS = tuple(base.FIXED_LIMITS)
DEPTHS = np.array(list(base.FIXED_LIMITS.values()))
BUDGETS = ("fixed16", "fixed32")
PENALTIES = (0.0, .01, .02, .04, .08, .16, .32, .64, 1.28, 2.56, 5.12)
RIDGE = 8.0
DEV_SALT = "qasper-adaptive-depth-v1"


def features(scores):
    ranked = np.sort(scores)[::-1]
    if not len(ranked):
        raise ValueError("A request needs at least one paragraph")
    positions = np.minimum([0, 4, 5, 8, 16, 32], len(ranked) - 1)
    best, fifth, sixth, ninth, seventeenth, thirty_third = ranked[positions]
    return np.array([np.log1p(len(ranked)), best, fifth, best-fifth, fifth-sixth,
                     fifth-ninth, fifth-seventeenth, fifth-thirty_third])


def partition(paper_id):
    hashed = sha256(f"{DEV_SALT}:split:{paper_id}".encode()).digest()
    return "calibration" if int.from_bytes(hashed[:4], "big") % 4 == 0 else "fit"


def fit(rows):
    labeled = [row for row in rows if row["quality"] is not None]
    x = np.array([row["features"] for row in labeled])
    y = np.array([row["quality"] for row in labeled])
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-9)
    scaled = (x - center) / scale
    coefficients = np.linalg.solve(scaled.T @ scaled + RIDGE * np.eye(x.shape[1]), scaled.T @ y)
    return dict(center=center, scale=scale, coefficients=coefficients, intercept=y.mean(0))


def choose(model, x, n, penalty):
    estimates = model["intercept"] + (x - model["center"]) / model["scale"] @ model["coefficients"]
    cost = np.minimum(DEPTHS, n)
    return int(np.argmax(estimates - penalty * cost / DEPTHS[-1]))


def calibrate(calibration, model, baseline):
    target = mean(min(row["paragraphs"], base.FIXED_LIMITS[baseline]) for row in calibration)
    trials = []
    for penalty in PENALTIES:
        chosen = [choose(model, row["features"], row["paragraphs"], penalty) for row in calibration]
        costs = [min(row["paragraphs"], int(DEPTHS[i])) for i, row in zip(chosen, calibration, strict=True)]
        quality = [row["quality"][i] for i, row in zip(chosen, calibration, strict=True)
                   if row["quality"] is not None]
        trials.append(dict(penalty=penalty, checks=mean(costs), complete=mean(quality)))
    valid = [trial for trial in trials if trial["checks"] <= target]
    if not valid:
        raise ValueError("No calibrated policy meets the fixed-depth check budget")
    winner = max(valid, key=lambda trial: (trial["complete"], -trial["checks"], -trial["penalty"]))
    baseline_index = ACTIONS.index(baseline)
    baseline_complete = mean(row["quality"][baseline_index] for row in calibration if row["quality"] is not None)
    return dict(target_checks=target, baseline_complete=baseline_complete, selected=winner, grid=trials)


def selected_dev(dataset):
    def key(qa):
        return sha256(f"{DEV_SALT}:{qa['question_id']}".encode()).digest()
    return [(pid, min(record["qas"], key=key)) for pid, record in sorted(dataset.items())]


def cases(dataset, selection, encoder):
    meta, _, vectors, offline = base.offline({pid: dataset[pid] for pid, _ in selection}, encoder)

    def iterate():
        for pid, qa in selection:
            paragraphs, sections, lo, hi = meta[pid]
            context = [f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)]
            bank = np.ascontiguousarray(vectors[lo:hi], dtype=np.float64)
            yield pid, qa, paragraphs, context, bank

    return iterate(), offline


def build_supervision(iterator, encoder, reader):
    rows = []
    for pid, qa, paragraphs, context, bank in iterator:
        question = qa["question"]
        cosine = bank @ np.asarray(encoder.encode([base.query_input(question)])[0], dtype=np.float64)
        gold = base.evidence_sets(qa, paragraphs)
        labels = None
        if gold:
            cache = {}

            def score(indices):
                key = tuple(indices)
                if key not in cache:
                    cache[key] = reader.score(question, context, indices, task=base.TASK)
                return cache[key]

            labels = []
            # Reuse identical reranker blocks across counterfactual actions in training only.
            for action in ACTIONS:
                order, _, _ = base.rank(cosine, score, action)
                labels.append(base.quality([paragraphs[index] for index in order], gold)["complete_evidence"])
        rows.append(dict(paper_id=pid, features=features(cosine), paragraphs=len(paragraphs), quality=labels))
        if len(rows) % 32 == 0:
            print("Development papers:", len(rows), flush=True)
    return rows


def measure(iterator, encoder, reader, model, policies):
    rows = []
    arms = (*base.ARMS, *(f"adaptive{budget.removeprefix('fixed')}" for budget in BUDGETS))
    for i, (pid, qa, paragraphs, context, bank) in enumerate(iterator):
        question = qa["question"]
        gold = base.evidence_sets(qa, paragraphs)
        offset = i % len(arms)
        results = {}
        for arm in arms[offset:] + arms[:offset]:
            started = perf_counter()
            cosine = bank @ np.asarray(encoder.encode([base.query_input(question)])[0], dtype=np.float64)
            action = arm
            if arm.startswith("adaptive"):
                policy = policies[arm.replace("adaptive", "fixed")]["selected"]
                action = ACTIONS[choose(model, features(cosine), len(paragraphs), policy["penalty"])]
            order, checked, clipped = base.rank(
                cosine, lambda ids: reader.score(question, context, ids, task=base.TASK), action)
            ms = (perf_counter() - started) * 1000
            results[arm] = dict(order=order, checked=checked, truncated=clipped, ms=ms, action=action,
                                human_evidence=base.quality([paragraphs[index] for index in order], gold))
        if results["certified"]["order"] != results["exhaustive"]["order"]:
            raise ValueError("Certified output differs from exhaustive")
        for arm, result in results.items():
            if arm.startswith("adaptive") and result["order"] != results[result["action"]]["order"]:
                raise ValueError("Adaptive action did not reproduce its fixed arm")
        rows.append(dict(paper_id=pid, question_id=qa["question_id"], paragraphs=len(paragraphs), arms=results))
        if len(rows) % 16 == 0:
            print("Test papers:", len(rows), flush=True)
    return rows, arms


def report(rows, arms):
    evaluated = [row for row in rows if row["arms"]["dense"]["human_evidence"] is not None]
    metrics = {}
    for arm in arms:
        metrics[arm] = dict(
            mean_checks=mean(row["arms"][arm]["checked"] for row in rows),
            p95_ms=float(np.quantile([row["arms"][arm]["ms"] for row in rows], .95)),
            complete=mean(row["arms"][arm]["human_evidence"]["complete_evidence"] for row in evaluated),
            f1=mean(row["arms"][arm]["human_evidence"]["f1"] for row in evaluated),
            actions={action: sum(row["arms"][arm]["action"] == action for row in rows)
                     for action in (*ACTIONS, "certified", "exhaustive")},
        )
        if arm.startswith("adaptive"):
            baseline = arm.replace("adaptive", "fixed")
            metrics[arm]["complete_delta_vs_budget_baseline"] = clustered_paired_interval([
                row["arms"][arm]["human_evidence"]["complete_evidence"]
                - row["arms"][baseline]["human_evidence"]["complete_evidence"] for row in evaluated
            ], [row["paper_id"] for row in evaluated])
    return dict(test_queries=len(rows), test_evaluable=len(evaluated), arms=metrics)


def run(data, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output path and a Slurm allocation")
    development = base.read_data(data, "dev")
    test = base.read_data(data, "test")
    if development.keys() & test.keys():
        raise ValueError("Development and test paper IDs overlap")
    embedding_info = base.verify_checkpoint(embedding)
    if embedding_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = base.QwenTorchEmbedding(embedding, max_length=512, batch_size=32)
    if not encoder.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    if reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473":
        raise ValueError("Unexpected reranker checkpoint")
    encoder.encode([base.query_input("What method is used?")])
    reader.score("What method is used?", ["The method is linear regression."] * base.BLOCK,
                 range(base.BLOCK), task=base.TASK)
    dev_iterator, dev_offline = cases(development, selected_dev(development), encoder)
    supervision = build_supervision(dev_iterator, encoder, reader)
    training = [row for row in supervision if partition(row["paper_id"]) == "fit"]
    calibration = [row for row in supervision if partition(row["paper_id"]) == "calibration"]
    model = fit(training)
    policies = {budget: calibrate(calibration, model, budget) for budget in BUDGETS}
    test_iterator, test_offline = cases(test, base.panel(test), encoder)
    rows, arms = measure(test_iterator, encoder, reader, model, policies)
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=encoder.torch.cuda.get_device_name(),
        data_assets=base.ASSETS, script_sha256=base.digest(Path(__file__)),
        ranker_script_sha256=base.digest(Path(base.__file__)), embedding=embedding_info, reranker=reader.provenance,
        offline=dict(development=dev_offline, test=test_offline),
        partitions=dict(fit=len(training), fit_evaluable=sum(r["quality"] is not None for r in training),
                        calibration=len(calibration), calibration_evaluable=sum(r["quality"] is not None for r in calibration)),
        model={key: value.tolist() for key, value in model.items()}, policies=policies, report=report(rows, arms),
        rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
