"""Test query-level verification routing on an independent judged paper corpus."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins, operating_point
from experiments.csfcube.facet_fallback_routing import fit_router, hashed, priorities, signals, utility
from experiments.csfcube.live_facet_gate import FacetIndex, act
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("dense4", "compact4", "cosine", "random", "margin", "learned")
TARGET = .10
TASK = "Decide whether the candidate paper is strongly relevant to the research described by the query paper."


class RelishIndex(FacetIndex):
    def __init__(self, corpus, questions, encoder):
        def view(doc, head=False):
            title, sentences = corpus[doc]
            return title + "\n" + " ".join(sentences[:3] if head else sentences)

        docs = sorted({doc for q in questions for doc in q["grades"]})
        self.full_pos = {doc: i for i, doc in enumerate(docs)}
        self.facet_pos = {(doc, "relevance"): i for i, doc in enumerate(docs)}
        self.compact = {(doc, "relevance"): view(doc, True) for doc in docs}
        self.full = encoder.encode([view(doc) for doc in docs])
        self.facet = encoder.encode(list(self.compact.values()))
        self.queries = {(q["query_id"], "relevance"): tuple(
            f"Find scientific research closely related to this paper.\n{view(q['query_id'], head)}"
            for head in (False, True)
        ) for q in questions}

    def check(self, reader, text, documents, facet):
        return margins(reader, [(text, self.compact[doc, facet]) for doc in documents], TASK)


def retrieve(question, index, encoder, reader, with_checks, *, cosine_needed=True):
    focal, docs, similarity = index.nominate(question, encoder, "reuse" if cosine_needed else "none")
    cosine = similarity.get("reuse")
    saved = dict(focal=focal, docs=docs, cosine=cosine)
    for name in with_checks:
        method = "dense4" if name == "dense4" else "facet4"
        indexes, _ = act(method, cosine, {})
        scores, cropped = index.check(reader, focal, [docs[k] for k in indexes], question["facet"])
        saved[name] = (indexes, scores, cropped)
    return saved


def top(positions, strengths, bias):
    return [(int(pos), float(value)) for pos, value in zip(positions, strengths, strict=True) if value > bias][:2]


def course(arm, question, sample, thresholds, model, routes):
    if arm in ("dense4", "compact4", "cosine"):
        return arm
    score = priorities(signals(sample["cosine"], thresholds), hashed(question), model)
    rule = "hash" if arm == "random" else arm
    return "compact4" if score[rule] > routes[arm] else "cosine"


def scaled(sample, course, thresholds):
    if course == "cosine":
        positions = range(len(sample["cosine"]))
        return positions, (sample["cosine"] - thresholds["cosine"]) / thresholds["cosine_scale"]
    positions, values, _ = sample[course]
    return positions, (values - thresholds["reader"]) / thresholds["reader_scale"]


def labels(question, sample, picks):
    return [question["grades"][sample["docs"][index]] for index, _ in picks]


def fit(index, encoder, reader, questions):
    samples = [retrieve(question, index, encoder, reader, ("dense4", "compact4")) for question in questions]
    grade = lambda question, rows, positions: [question["grades"][rows["docs"][i]] for i in positions]
    cosine = np.concatenate([row["cosine"] for row in samples])
    cos_labels = np.array([g for q, row in zip(questions, samples, strict=True)
                           for g in grade(q, row, range(len(row["docs"])))])
    evidence = np.concatenate([row["dense4"][1] for row in samples])
    ev_labels = np.array([g for q, row in zip(questions, samples, strict=True)
                          for g in grade(q, row, row["dense4"][0])])
    thresholds = dict(cosine=operating_point(cosine, cos_labels), reader=operating_point(evidence, ev_labels),
                      cosine_scale=float(cosine.std()), reader_scale=float(evidence.std()))
    features, gains = [], []
    for question, row in zip(questions, samples, strict=True):
        features.append(signals(row["cosine"], thresholds))
        fast = labels(question, row, top(*scaled(row, "cosine", thresholds), 0))
        checked = labels(question, row, top(*scaled(row, "compact4", thresholds), 0))
        gains.append(utility(checked) - utility(fast))
    model = fit_router(np.stack(features), np.asarray(gains))
    priorities_fit = [priorities(vector, hashed(q), model) for vector, q in zip(features, questions, strict=True)]
    routes = {method: float(np.quantile([row[rule] for row in priorities_fit], .70))
              for method, rule in (("random", "hash"), ("margin", "margin"), ("learned", "learned"))}
    return thresholds, model, routes


def risk_fit(index, encoder, reader, questions, thresholds, model, routes):
    samples = [retrieve(question, index, encoder, reader, ("dense4", "compact4")) for question in questions]
    risk = {}
    for arm in ARMS:
        candidates = [scaled(sample, course(arm, question, sample, thresholds, model, routes), thresholds)
                      for question, sample in zip(questions, samples, strict=True)]
        shifts = {float(value) for _, scores in candidates for value in scores if value > 0} | {0.}
        for shift in sorted(shifts):
            bad = sum(0 in labels(q, row, top(*candidate, shift))
                      for q, row, candidate in zip(questions, samples, candidates, strict=True))
            if bad <= TARGET * len(questions):
                risk[arm] = dict(shift=shift, adverse_fit=bad)
                break
    return risk


def online(questions, index, encoder, reader, thresholds, model, routes, risk, offset):
    values = {arm: [] for arm in ARMS}
    for j, question in enumerate([questions[0], *questions]):
        step = (j + offset) % len(ARMS)
        for arm in ARMS[step:] + ARMS[:step]:
            clock = perf_counter()
            sample = retrieve(question, index, encoder, reader, (), cosine_needed=arm != "dense4")
            branch = course(arm, question, sample, thresholds, model, routes)
            if branch != "cosine":
                method = "dense4" if branch == "dense4" else "facet4"
                positions, _ = act(method, sample["cosine"], thresholds)
                scores, clipped = index.check(reader, sample["focal"], [sample["docs"][i] for i in positions], question["facet"])
                sample[branch] = (positions, scores, clipped)
            picks = top(*scaled(sample, branch, thresholds), risk[arm]["shift"])
            json.dumps({"matches": [sample["docs"][i] for i, _ in picks]})
            elapsed = (perf_counter() - clock) * 1000
            if j:
                grades = labels(question, sample, picks)
                values[arm].append(dict(issued=list(zip(grades, (v for _, v in picks), strict=True)),
                                        checked=branch != "cosine", ms=elapsed))
    return values


def match_risk(rows, maximum):
    shifts = sorted({score for row in rows for _, score in row["issued"]})
    options = [[grade for grade, _ in row["issued"]] for row in rows]
    if sum(0 in row for row in options) <= maximum:
        return options
    for cutoff in shifts:
        options = [[grade for grade, strength in row["issued"] if strength > cutoff] for row in rows]
        if sum(0 in row for row in options) <= maximum:
            return options
    raise AssertionError("Empty issuance must meet the risk budget")


def summarize(values, questions):
    def counts(grades):
        aggregate = Counter(g for row in grades for g in row)
        return dict(strong=sum(any(g >= 2 for g in row) for row in grades),
                    adverse=sum(0 in row for row in grades), issued=[aggregate[g] for g in (0, 1, 2)])

    raw = {arm: [[g for g, _ in row["issued"]] for row in rows] for arm, rows in values.items()}
    budget = min(counts(raw[arm])["adverse"] for arm in ("dense4", "learned"))
    matched = {arm: match_risk(rows, budget) for arm, rows in values.items()}
    groups = [question["query_id"] for question in questions]
    report = {}
    for arm, rows in values.items():
        delta = [int(any(g >= 2 for g in one)) - int(any(g >= 2 for g in other))
                 for one, other in zip(raw[arm], raw["dense4"], strict=True)]
        report[arm] = dict(
            **counts(raw[arm]), matched_adverse=counts(matched[arm]), checks=sum(row["checked"] for row in rows),
            p50_p95_ms=np.quantile([row["ms"] for row in rows], [.50, .95]).tolist(),
            raw_paired_delta=clustered_paired_interval(delta, groups),
        )
    return dict(arms=report, held_adverse_budget=budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel_path", "manifest", "checkpoint", "verifier", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    panel_path, manifest, checkpoint, verifier, output = (
        args.panel_path, args.manifest, args.checkpoint, args.verifier, args.output
    )
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    panel = json.loads(panel_path.read_bytes())
    if digest(panel_path) != json.loads(manifest.read_bytes())["panel_sha256"]:
        raise ValueError("RELISH panel changed")
    embed = verify_checkpoint(checkpoint)
    encoder = QwenTorchEmbedding(checkpoint)
    encoder.model.config.use_cache = False
    reader = Reranker(verifier)
    schedules = panel["panels"]
    index = RelishIndex(panel["documents"], [question for rows in schedules.values() for question in rows], encoder)
    thresholds, model, routes = fit(index, encoder, reader, schedules["fit"])
    risk = risk_fit(index, encoder, reader, schedules["risk"], thresholds, model, routes)
    runs = [summarize(online(schedules["held"], index, encoder, reader, thresholds, model, routes, risk, shift),
                      schedules["held"]) for shift in (0, 1)]
    write_json(output, dict(runs=runs, thresholds=thresholds, routes=routes, risk=risk, gpu=torch.cuda.get_device_name(),
                            panel_sha256=digest(panel_path), script_sha256=digest(Path(__file__)),
                            embedding_revision=embed["revision"], reranker_revision=reader.provenance["revision"]))


if __name__ == "__main__":
    main()
