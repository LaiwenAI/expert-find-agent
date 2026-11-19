"""Test evidence completion and reviewer relevance under matched citation counts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from experiments.retrieval import cof_followup_value as prior


REFERENCE = "9b62a8e359898d6178ab8d0915d3602adf9cdc7dd9d128d83dd231cc686be9b1"
CONTROLS = ("round_robin", "one_hit", "margin", "learned")
ARMS = (*CONTROLS, "completion", "factorized")


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -30., 30.)))


def logistic(x, y):
    w = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = .01
    for _ in range(12):
        p = sigmoid(x @ w)
        gradient = x.T @ (p - y) + penalty @ w
        hessian = x.T @ (x * (p * (1. - p))[:, None]) + penalty
        w -= np.linalg.solve(hessian, gradient)
    return w


def reviewer_features(request, observed, state):
    support, confirmed, _ = state
    rows = []
    for i, (_, docs) in enumerate(request.hits):
        margins = np.clip([observed[d] for d in docs if d in observed], -8., 8.) / 8.
        rows.append([1., i / len(request.hits), i < prior.TOP, len(docs) / prior.DEPTH,
                     len(margins) / prior.DEPTH, support[i] / prior.DEPTH,
                     float(max(margins, default=-1.)), float(sum(margins) / max(1, len(margins))),
                     confirmed[i]])
    return np.asarray(rows)


def fit(requests):
    source_x, source_y, reviewer_x, reviewer_y = [], [], [], []
    for request in requests:
        observed = request.initial_values()
        for _ in range(prior.FOLLOWUP):
            options = request.ordered_options(observed)
            if not options:
                break
            state = request.ranks(observed)
            source_x.extend(request.features(doc, observed, state) for doc in options)
            source_y.extend(request.margins[doc] >= 0 for doc in options)
            reviewer_x.extend(reviewer_features(request, observed, state))
            reviewer_y.extend(request.labels[owner] == 1 for owner, _ in request.hits)
            next_doc = request.choose("one_hit", observed, options, None)
            observed[next_doc] = request.margins[next_doc]
    xs, xr = np.stack(source_x), np.stack(reviewer_x)
    ys, yr = np.asarray(source_y, dtype=float), np.asarray(reviewer_y, dtype=float)
    return logistic(xs, ys), logistic(xr, yr), dict(
        source_examples=len(ys), source_positives=int(ys.sum()),
        reviewer_examples=len(yr), reviewer_positives=int(yr.sum()),
    )


def citation_slots(state):
    _, confirmed, order = state
    return {rank for rank in order if confirmed[rank]}


def choose(request, observed, options, method, models):
    model_source, model_reviewer = models
    state = request.ranks(observed)
    covered = citation_slots(state)
    source_p = sigmoid(np.stack([request.features(doc, observed, state) for doc in options]) @ model_source)
    deltas = [request.ranks(observed | {doc: 0.}) for doc in options]
    coverage = np.asarray([len(citation_slots(after)) - len(covered) for after in deltas])
    maximum = int(coverage.max())
    if maximum <= 0:
        return request.choose("one_hit", observed, options, None)
    allowed = np.where(coverage == maximum)[0]
    if method == "completion":
        return options[int(allowed[np.argmax(source_p[allowed])])]
    relevance = sigmoid(reviewer_features(request, observed, state) @ model_reviewer)
    order_now = set(state[2])

    def utility(i):
        after = deltas[i]
        order_after = set(after[2])
        cited_after = citation_slots(after)
        rank_delta = sum(relevance[k] for k in order_after - order_now) - sum(
            relevance[k] for k in order_now - order_after)
        cite_delta = sum(relevance[k] for k in cited_after - covered) - sum(
            relevance[k] for k in covered - cited_after)
        return source_p[i] * (rank_delta + cite_delta)

    return options[max(allowed, key=utility)]


def follow(request, method, models, direct):
    if method in CONTROLS:
        return prior.select_documents(request, method, direct)
    observed = request.initial_values()
    clock = 0.
    for _ in range(prior.FOLLOWUP):
        options = request.ordered_options(observed)
        if not options:
            break
        started = perf_counter()
        path = choose(request, observed, options, method, models)
        clock += (perf_counter() - started) * 1000
        observed[path] = request.margins[path]
    return observed, clock


def comparison(request, target, base):
    common = min(len(target), len(base))
    candidate = {request.hits[i][0] for i in sorted(target)[:common]}
    reference = {request.hits[i][0] for i in sorted(base)[:common]}
    gained, dropped = candidate - reference, reference - candidate
    labels = request.labels
    positive = sum(labels.get(r) == 1 for r in candidate)
    base_positive = sum(labels.get(r) == 1 for r in reference)
    authored_delta = positive - base_positive
    return dict(slots=common, positive=positive, base_positive=base_positive,
                negative=sum(labels.get(r) == 0 for r in candidate),
                base_negative=sum(labels.get(r) == 0 for r in reference),
                unknown=len(candidate - labels.keys()), base_unknown=len(reference - labels.keys()),
                lower=authored_delta - len(dropped - labels.keys()),
                upper=authored_delta + len(gained - labels.keys()))


def evaluate(requests, models, direct):
    results, matched_counts = defaultdict(list), defaultdict(list)
    for request in requests:
        expected = min(len(request.margins), len(request.initial) + prior.FOLLOWUP)
        selected = {arm: follow(request, arm, models, direct) for arm in ARMS}
        for arm, (observed, elapsed) in selected.items():
            if len(observed) != expected:
                raise AssertionError("Policies did not spend an identical number of distinct reads")
            results[arm].append(dict(request.quality(observed), reads=expected, decision_ms=elapsed))
            for reference in ("one_hit", "completion"):
                a = citation_slots(request.ranks(observed))
                b = citation_slots(request.ranks(selected[reference][0]))
                matched_counts[arm, reference].append(comparison(request, a, b))

    output = {}
    for arm, rows in results.items():
        sample = dict(Counter({key: sum(row[key] for row in rows) for key in
                              ("positive", "judged", "cited_positive", "cited_negative", "cited_unknown")}),
                      mean_reads=mean(r["reads"] for r in rows),
                      decision_p95_ms=float(np.quantile([r["decision_ms"] for r in rows], .95)))
        for reference in ("one_hit", "completion"):
            matches = matched_counts[arm, reference]
            sample[f"matched_vs_{reference}"] = dict(
                {key: sum(row[key] for row in matches) for key in matches[0]},
                delta=prior.paired_interval([(row["positive"] - row["base_positive"]) / prior.TOP for row in matches]),
            )
        output[arm] = sample
    return output


def run(embedding, reranker, reference_file, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or prior.digest(reference_file) != REFERENCE):
        raise ValueError("Require the pinned comparison and a fresh result on Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = prior.verify_checkpoint(embedding)
    encoder = prior.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = prior.Reranker(reranker)
    baseline = json.loads(reference_file.read_bytes())
    if (baseline["embedding_revision"] != provenance["revision"]
            or baseline["reranker_revision"] != verifier.provenance["revision"]):
        raise ValueError("Changed checkpoints")
    scopes = {}
    all_titles = set()
    models = direct = train_info = None
    for dataset in (*prior.DATASETS, "NIPS"):
        kwargs = {"select_queries": lambda _dataset, ids, _texts: ids} if dataset == "NIPS" else {}
        root, data, experts, sources, truncated, titles = prior.collect(dataset, encoder, verifier, provenance, **kwargs)
        if titles & all_titles:
            raise ValueError(f"Duplicate query title in {dataset}")
        all_titles.update(titles)
        if dataset == "SIGIR":
            data = sorted(data, key=lambda row: sha256(f"cof-followup-v1:{row[0]}".encode()).digest())
            split = 2 * len(data) // 3
            train = data[:split]
            ids = sha256("\n".join(q for q, _ in train).encode()).hexdigest()
            if ids != baseline["training_ids_sha256"]:
                raise ValueError("Changed fitted query IDs")
            models = fit([r for _, r in train])
            direct = prior.fit([r for _, r in train])[0]
            train_info = dict(hashes=ids, queries=len(train), **models[2])
            data = data[split:]
        result = evaluate([r for _, r in data], models[:2], direct)
        cohort = sha256("\n".join(q for q, _ in data).encode()).hexdigest()
        if dataset in baseline["report"]:
            expected = baseline["report"][dataset]
            if cohort != expected["qids_sha256"]:
                raise ValueError("Changed replay queries")
            for arm in CONTROLS:
                for key in ("positive", "judged", "cited_positive", "cited_negative", "cited_unknown"):
                    if result[arm][key] != expected["report"][arm][key]:
                        raise ValueError(f"Failed baseline reproduction on {dataset} {arm} {key}")
        scopes[dataset] = dict(queries=len(data), experts=experts, sources=sources,
                               qids_sha256=cohort, labels_sha256=prior.digest(root / f"{dataset}_queries_test_hard.json"),
                               truncated_pairs=truncated, report=result)

    prior.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=prior.digest(Path(__file__)),
        reference_sha256=REFERENCE, embedding_revision=provenance["revision"],
        reranker_revision=verifier.provenance["revision"], training=train_info,
        source_weights=models[0].tolist(), relevance_weights=models[1].tolist(), report=scopes,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("embedding", "reranker", "reference_file", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
