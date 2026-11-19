"""Test a cost-calibrated fallback for ambiguous author contribution claims."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from experiments.credit import compile_legacy_notes as indexed
from experiments.credit import legacy_contribution_text as legacy


REFERENCE = "a9cc9ca5483584846953fb337dd30406cba36f1e84d35b416caa57d0fac91c9e"
FEATURES = {"character": (0,), "semantic": (1,), "agreement": (0, 1)}
LOW = (None, .05, .10, .20, .30, .40)
HIGH = (.60, .70, .80, .90, .95, None)


@dataclass(frozen=True)
class Gate:
    features: tuple
    low: tuple
    high: tuple


def score_pair(row, entry, version, encoder, feature_ids):
    _, _, tasks, *authors = row
    words = [legacy.PHRASES[name][version] for name in tasks]
    outputs = []
    for feature in feature_ids:
        if feature == 0:
            scores = np.stack([
                indexed.grams_score(indexed.char_grams(phrase), entry["grams"]) for phrase in words
            ])
        else:
            encoded = encoder.encode([f"Instruct: {indexed.TASK}\nQuery: {word}" for word in words])
            scores = encoded @ entry["vectors"].T
        outputs.append([indexed.weakest(scores, entry["clauses"], owner, True, -float(feature))[0]
                        for owner in authors])
    return np.asarray(outputs).T


def reader_pair(row, version, engines):
    reader, tokenizer, digits = engines
    doi, text, tasks, good, other = row
    order = (other, good) if legacy.stable(doi)[0] % 2 else (good, other)
    query = [f"Target author initials: {owner}\nRequired contributions: "
             + "; ".join(legacy.PHRASES[task][version] for task in tasks) for owner in order]
    values = indexed.judge(reader, tokenizer, digits, "Apply the decision separately to this author.",
                           [f"{q}\n\n{text}" for q in query])
    return np.asarray(values if order[0] == good else values[::-1])


def first_stage(scores, gate):
    positive = (scores > np.asarray(gate.high)).all(axis=-1)
    negative = (scores <= np.asarray(gate.low)).all(axis=-1)
    pending = ~(positive | negative)
    return positive, pending


def choose_gates(scores, truth, times):
    base = dict(gate=None, calls=len(truth), mean_ms=mean(times["reader"]),
                supported=int(truth[:, 0].sum()), errors=int(truth[:, 1].sum()))
    choices = {}
    for name, columns in FEATURES.items():
        data = scores[:, :, columns]
        full = data.reshape(-1, len(columns))
        best = base
        feasible = 0
        fastest_choice = None
        for a in LOW:
            low = tuple(np.full(len(columns), -np.inf) if a is None else np.quantile(full, a, axis=0))
            for b in HIGH:
                high = tuple(np.full(len(columns), np.inf) if b is None else np.quantile(full, b, axis=0))
                gate = Gate(columns, low, high)
                accepted, pending = first_stage(data, gate)
                prediction = accepted | (pending & truth)
                keep, errors = prediction.sum(axis=0)
                if keep < base["supported"] or errors > base["errors"]:
                    continue
                feasible += 1
                calls = pending.any(axis=1)
                estimated = np.asarray(times[name]) + np.asarray(times["reader"]) * calls
                candidate = dict(gate=gate, calls=int(calls.sum()), mean_ms=float(estimated.mean()),
                                 supported=int(keep), errors=int(errors))
                criterion = lambda item: (item["mean_ms"], item["calls"], item["errors"], -item["supported"])
                if fastest_choice is None or criterion(candidate) < criterion(fastest_choice):
                    fastest_choice = candidate
                if criterion(candidate) < criterion(best):
                    best = candidate
        choices[name] = dict(selected=best, best_gate_even_if_slower=fastest_choice, feasible=feasible)
    return base, choices


def stats(predictions, times, calls, base):
    z = np.asarray(predictions, dtype=bool)
    target = np.asarray(base, dtype=bool)
    return dict(supported_accepted=int(z[:, 0].sum()), wrong_owner_accepted=int(z[:, 1].sum()),
                exact_pairs=int((z[:, 0] & ~z[:, 1]).sum()),
                reader_calls=sum(calls), mean_ms=mean(times), p95_ms=float(np.quantile(times, .95)),
                support_delta=indexed.paired_interval(z[:, 0].astype(int) - target[:, 0]),
                wrong_owner_delta=indexed.paired_interval(z[:, 1].astype(int) - target[:, 1]))


def sample(rows, catalog, version, encoder, engines):
    scores, reader_scores = [], []
    timings = {"reader": [], **{name: [] for name in FEATURES}}
    for row in rows:
        doc = catalog[row[0]]
        for name, column in FEATURES.items():
            t0 = perf_counter()
            s = score_pair(row, doc, version, encoder, column)
            timings[name].append((perf_counter() - t0) * 1000)
            if name == "agreement":
                scores.append(s)
        t0 = perf_counter()
        reader_scores.append(reader_pair(row, version, engines))
        timings["reader"].append((perf_counter() - t0) * 1000)
    return np.asarray(scores), np.asarray(reader_scores), timings


def live(rows, catalog, encoder, engines, gate, threshold):
    outcomes, times, calls = [], [], []
    for row in rows:
        started = perf_counter()
        if gate is None:
            calls.append(True)
            prediction = reader_pair(row, 1, engines) > threshold
        else:
            scores = score_pair(row, catalog[row[0]], 1, encoder, gate.features)
            prediction, pending = first_stage(scores, gate)
            calls.append(bool(pending.any()))
            if calls[-1]:
                prediction |= pending & (reader_pair(row, 1, engines) > threshold)
        times.append((perf_counter() - started) * 1000)
        outcomes.append(prediction)
    return outcomes, times, calls


def lazy_agreement(rows, catalog, encoder, engines, gate, threshold):
    if gate.features != (0, 1):
        raise ValueError("Expected a two-signal agreement gate")
    character = Gate((0,), gate.low[:1], gate.high[:1])
    semantic = Gate((1,), gate.low[1:], gate.high[1:])
    outcomes, times, calls = [], [], []
    semantic_calls = 0
    for row in rows:
        started = perf_counter()
        entry = catalog[row[0]]
        lexical, lexical_pending = first_stage(score_pair(row, entry, 1, encoder, (0,)), character)
        if lexical_pending.any():
            # At least one candidate cannot pass agreement, so the reader is mandatory.
            reader = reader_pair(row, 1, engines) > threshold
            calls.append(True)
            can_change = ~lexical_pending & (lexical != reader)
            if can_change.any():
                vector, vector_pending = first_stage(score_pair(row, entry, 1, encoder, (1,)), semantic)
                semantic_calls += 1
                override = can_change & ~vector_pending & (vector == lexical)
                reader = np.where(override, lexical, reader)
            prediction = reader
        else:
            vector, vector_pending = first_stage(score_pair(row, entry, 1, encoder, (1,)), semantic)
            semantic_calls += 1
            pending = vector_pending | (vector != lexical)
            calls.append(bool(pending.any()))
            prediction = lexical
            if calls[-1]:
                reader = reader_pair(row, 1, engines) > threshold
                prediction = np.where(pending, reader, lexical)
        outcomes.append(prediction)
        times.append((perf_counter() - started) * 1000)
    return outcomes, times, calls, semantic_calls


def serializable(choice):
    if choice is None:
        return None
    gate = choice["gate"]
    params = None if gate is None else dict(
        features=list(gate.features), reject_below=[float(x) if np.isfinite(x) else None for x in gate.low],
        accept_above=[float(x) if np.isfinite(x) else None for x in gate.high],
    )
    return dict(choice, gate=params)


def run(snapshot, embedding, reader, reference, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or indexed.digest(reference) != REFERENCE):
        raise ValueError("Require a pinned baseline, new output and Slurm CUDA")
    old = json.loads(reference.read_text())
    data = legacy.prepare(snapshot)
    cal, test = data["calibration"][0], data["test"][0]
    if (len(cal), len(test)) != (old["calibration_papers"], old["test_papers"]):
        raise ValueError("Changed attribution cohort")
    provenance = indexed.verify_checkpoint(embedding)
    if provenance["revision"] != old["embedding_revision"]:
        raise ValueError("Changed embedding revision")
    encoder = indexed.QwenTorchEmbedding(embedding, batch_size=8)
    encoder.model.config.use_cache = False
    corpus, build = indexed.catalog(data, encoder)
    tokenizer, model, reader_info = indexed.load_reader(reader)
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(item) != 1 for item in ids) or ids[0] == ids[1]:
        raise ValueError("Expected distinct single-token decisions")
    engines = (model, tokenizer, [item[0] for item in ids])
    sample(cal[:1], corpus, 0, encoder, engines)
    cal_scores, known, times = sample(cal, corpus, 0, encoder, engines)
    threshold = old["baselines"]["reader_4b"]["threshold"]
    truth = known > threshold
    baseline, routes = choose_gates(cal_scores, truth, times)
    base_out, base_times, base_calls = live(test, corpus, encoder, engines, None, threshold)
    reference_results = old["baselines"]["reader_4b"]["held_threshold"]
    if (sum(row[0] for row in base_out) != reference_results["supported_accepted"]
            or sum(row[1] for row in base_out) != reference_results["wrong_owner_accepted"]):
        raise ValueError("Changed 4B decisions at the published cutoff")
    report = {"reader": stats(base_out, base_times, base_calls, base_out)}
    for route, config in routes.items():
        # Always report the actually selected path. When it selects the fixed
        # reader, retain the faster-candidate-family result as a diagnostic.
        chosen = config["selected"]
        if chosen["gate"] is None:
            report[route] = report["reader"].copy()
        else:
            report[route] = stats(*live(test, corpus, encoder, engines, chosen["gate"], threshold), base_out)
        alternative = config["best_gate_even_if_slower"]
        if chosen["gate"] is None and alternative is not None:
            report[route]["forced_gate_diagnostic"] = stats(
                *live(test, corpus, encoder, engines, alternative["gate"], threshold), base_out)
    agreement = routes["agreement"]["selected"]
    aggressive = agreement if agreement["gate"] else routes["agreement"]["best_gate_even_if_slower"]
    if aggressive is not None:
        eager, eager_ms, eager_calls = live(test, corpus, encoder, engines, aggressive["gate"], threshold)
        lazy, lazy_ms, lazy_calls, vector_count = lazy_agreement(
            test, corpus, encoder, engines, aggressive["gate"], threshold)
        if not np.array_equal(eager, lazy) or eager_calls != lazy_calls:
            raise ValueError("Lazy scheduling changed agreement outputs or mandatory reader calls")
        report["eager_agreement_schedule"] = dict(stats(eager, eager_ms, eager_calls, base_out),
                                                  semantic_calls=len(test))
        report["lazy_agreement_schedule"] = dict(stats(lazy, lazy_ms, lazy_calls, base_out),
                                                  semantic_calls=vector_count, identical_to_eager=len(test))
    settings = {k: dict(selected=serializable(v["selected"]), feasible=v["feasible"],
                         best_gate_even_if_slower=serializable(v["best_gate_even_if_slower"]))
                for k, v in routes.items()}
    indexed.write_json(output, dict(
        script_sha256=indexed.digest(Path(__file__)), reference_sha256=REFERENCE,
        capture_sha256=indexed.digest(snapshot / "capture.json"), embedding_revision=provenance["revision"],
        reader_revision=reader_info["revision"], gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
        calibration_papers=len(cal), test_papers=len(test), compiled=build, reader_cutoff=threshold,
        calibration_reader=serializable(baseline), routes=settings, test=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("snapshot", "embedding", "reader", "reference", "output"):
        parser.add_argument(item, type=Path)
    run(**vars(parser.parse_args()))
