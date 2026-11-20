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
from experiments.csfcube.facet_fallback_routing import fit_router, hashed, priorities, signals
from experiments.csfcube.live_facet_gate import FacetIndex, act
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("dense4", "compact4", "cosine", "random", "margin", "learned")
STRICT_ARMS = ("dense8", *ARMS)
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
        indexes = check_positions(name, cosine)
        scores, cropped = index.check(reader, focal, [docs[k] for k in indexes], question["facet"])
        saved[name] = (indexes, scores, cropped)
    return saved


def check_positions(branch, cosine):
    method = "facet4" if branch == "compact4" else branch
    return act(method, cosine, {})[0]


def adverse(grades, maximum_grade):
    return any(grade <= maximum_grade for grade in grades)


def top(positions, strengths, bias):
    return [(int(pos), float(value)) for pos, value in zip(positions, strengths, strict=True) if value > bias][:2]


def course(arm, question, sample, thresholds, model, routes):
    if arm in ("dense8", "dense4", "compact4", "cosine"):
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


def fit(index, encoder, reader, questions, adverse_grade=0):
    samples = [retrieve(question, index, encoder, reader, ("dense4", "compact4")) for question in questions]
    grade = lambda question, rows, positions: [question["grades"][rows["docs"][i]] for i in positions]
    cosine = np.concatenate([row["cosine"] for row in samples])
    cos_labels = np.array([g for q, row in zip(questions, samples, strict=True)
                           for g in grade(q, row, range(len(row["docs"])))])
    evidence = np.concatenate([row["dense4"][1] for row in samples])
    ev_labels = np.array([g for q, row in zip(questions, samples, strict=True)
                          for g in grade(q, row, row["dense4"][0])])
    thresholds = dict(cosine=operating_point(cosine, cos_labels > adverse_grade),
                      reader=operating_point(evidence, ev_labels > adverse_grade),
                      cosine_scale=float(cosine.std()), reader_scale=float(evidence.std()))
    features, gains = [], []
    for question, row in zip(questions, samples, strict=True):
        features.append(signals(row["cosine"], thresholds))
        fast = labels(question, row, top(*scaled(row, "cosine", thresholds), 0))
        checked = labels(question, row, top(*scaled(row, "compact4", thresholds), 0))
        gains.append(
            int(2 in checked) - int(2 in fast)
            - .5 * (int(adverse(checked, adverse_grade)) - int(adverse(fast, adverse_grade)))
        )
    model = fit_router(np.stack(features), np.asarray(gains))
    priorities_fit = [priorities(vector, hashed(q), model) for vector, q in zip(features, questions, strict=True)]
    routes = {method: float(np.quantile([row[rule] for row in priorities_fit], .70))
              for method, rule in (("random", "hash"), ("margin", "margin"), ("learned", "learned"))}
    return thresholds, model, routes


def risk_fit(index, encoder, reader, questions, thresholds, model, routes, arms=ARMS, adverse_grade=0):
    checks = [arm for arm in arms if arm in ("dense8", "dense4", "compact4")]
    samples = [retrieve(question, index, encoder, reader, checks) for question in questions]
    risk = {}
    for arm in arms:
        candidates = [scaled(sample, course(arm, question, sample, thresholds, model, routes), thresholds)
                      for question, sample in zip(questions, samples, strict=True)]
        shifts = {float(value) for _, scores in candidates for value in scores if value > 0} | {0.}
        for shift in sorted(shifts):
            bad = sum(adverse(labels(q, row, top(*candidate, shift)), adverse_grade)
                      for q, row, candidate in zip(questions, samples, candidates, strict=True))
            if bad <= TARGET * len(questions):
                risk[arm] = dict(shift=shift, adverse_fit=bad)
                break
    return risk


def online(questions, index, encoder, reader, thresholds, model, routes, risk, offset, arms=ARMS):
    values = {arm: [] for arm in arms}
    for j, question in enumerate([questions[0], *questions]):
        step = (j + offset) % len(arms)
        for arm in arms[step:] + arms[:step]:
            clock = perf_counter()
            sample = retrieve(question, index, encoder, reader, (), cosine_needed=arm not in ("dense4", "dense8"))
            branch = course(arm, question, sample, thresholds, model, routes)
            if branch != "cosine":
                positions = check_positions(branch, sample["cosine"])
                scores, clipped = index.check(reader, sample["focal"], [sample["docs"][i] for i in positions], question["facet"])
                sample[branch] = (positions, scores, clipped)
            positions, normalized = scaled(sample, branch, thresholds)
            picks = top(positions, normalized, risk[arm]["shift"])
            json.dumps({"matches": [sample["docs"][i] for i, _ in picks]})
            elapsed = (perf_counter() - clock) * 1000
            if j:
                grades = labels(question, sample, picks)
                all_scored = list(zip(labels(question, sample, [(i, 0) for i in positions]), normalized, strict=True))
                values[arm].append(dict(issued=list(zip(grades, (v for _, v in picks), strict=True)),
                                        checked=0 if branch == "cosine" else len(positions),
                                        normalized=all_scored, ms=elapsed))
    return values


def match_risk(rows, maximum, adverse_grade=0):
    shifts = sorted({score for row in rows for _, score in row["issued"]})
    options = [[grade for grade, _ in row["issued"]] for row in rows]
    if sum(adverse(row, adverse_grade) for row in options) <= maximum:
        return options
    for cutoff in shifts:
        options = [[grade for grade, strength in row["issued"] if strength > cutoff] for row in rows]
        if sum(adverse(row, adverse_grade) for row in options) <= maximum:
            return options
    raise AssertionError("Empty issuance must meet the risk budget")


def summarize(values, questions, risk, adverse_grade=0):
    def counts(grades):
        aggregate = Counter(g for row in grades for g in row)
        return dict(strong=sum(any(g >= 2 for g in row) for row in grades),
                    adverse=sum(adverse(row, adverse_grade) for row in grades), issued=[aggregate[g] for g in (0, 1, 2)])

    raw = {arm: [[g for g, _ in row["issued"]] for row in rows] for arm, rows in values.items()}
    controls = ("dense4", "learned")
    if adverse_grade == 1:
        controls += ("dense8", "cosine", "random")
    budget = min(counts(raw[arm])["adverse"] for arm in controls)
    matched = {arm: match_risk(rows, budget, adverse_grade) for arm, rows in values.items()}
    groups = [question["query_id"] for question in questions]
    report = {}
    for arm, rows in values.items():
        delta = [int(any(g >= 2 for g in one)) - int(any(g >= 2 for g in other))
                 for one, other in zip(raw[arm], raw["dense4"], strict=True)]
        report[arm] = dict(
            **counts(raw[arm]), matched_adverse=counts(matched[arm]),
            checks=sum(row["checked"] > 0 for row in rows), pairs=sum(row["checked"] for row in rows),
            raw_grade0_adverse=sum(0 in row for row in raw[arm]),
            p50_p95_ms=np.quantile([row["ms"] for row in rows], [.50, .95]).tolist(),
            raw_paired_delta=clustered_paired_interval(delta, groups),
        )
    if adverse_grade == 0:
        return dict(arms=report, held_adverse_budget=budget)

    def paired(focal, control):
        delta = [int(2 in a) - int(2 in b) for a, b in zip(focal, control, strict=True)]
        return clustered_paired_interval(delta, groups)

    comparisons = {
        name: dict(raw=paired(raw["learned"], raw[name]), matched=paired(matched["learned"], matched[name]))
        for name in ("dense4", "dense8", "cosine", "random")
    }
    cutoff = risk["learned"]["shift"]
    variants = {
        arm: [[grade for grade, score in row["normalized"] if score > cutoff][:2] for row in values[arm]]
        for arm in ("cosine", "compact4")
    }
    fast = np.array([(2 in row, adverse(row, adverse_grade)) for row in variants["cosine"]], dtype=int)
    slow = np.array([(2 in row, adverse(row, adverse_grade)) for row in variants["compact4"]], dtype=int)
    work = report["learned"]["checks"]
    rng = np.random.default_rng(20261121)
    draws = np.stack([fast.sum(axis=0) + (slow-fast)[rng.choice(len(fast), size=work, replace=False)].sum(axis=0)
                      for _ in range(2000)])
    return dict(arms=report, held_adverse_budget=budget, learned_vs=comparisons,
                same_budget_random_replay=dict(
                    checks=work, mean=draws.mean(axis=0).tolist(),
                    central_95=np.quantile(draws, [.025, .975], axis=0).T.tolist(),
                ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel_path", "manifest", "checkpoint", "verifier", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--adverse-grade", type=int, choices=(0, 1), default=0)
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
    arms = STRICT_ARMS if args.adverse_grade == 1 else ARMS
    thresholds, model, routes = fit(index, encoder, reader, schedules["fit"], args.adverse_grade)
    risk = risk_fit(index, encoder, reader, schedules["risk"], thresholds, model, routes, arms, args.adverse_grade)
    runs = [summarize(
        online(schedules["held"], index, encoder, reader, thresholds, model, routes, risk, shift, arms),
        schedules["held"], risk, args.adverse_grade,
    ) for shift in (0, 1)]
    write_json(output, dict(runs=runs, thresholds=thresholds, routes=routes, risk=risk, gpu=torch.cuda.get_device_name(),
                            adverse_grade=args.adverse_grade,
                            panel_sha256=digest(panel_path), script_sha256=digest(Path(__file__)),
                            embedding_revision=embed["revision"], reranker_revision=reader.provenance["revision"]))


if __name__ == "__main__":
    main()
