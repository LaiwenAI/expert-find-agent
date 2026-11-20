"""Measure a cached facet gate at the normal verification entry point."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins, operating_point
from experiments.csfcube.frozen_confirmation import FACETS, load
from experiments.csfcube.paired_facet_confirmation import TASK, key, render
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("dense8", "dense4", "facet4", "reject_only", "two_sided", "cosine_only",
        "reuse4", "reuse_reject", "reuse_cosine")
REUSE = {"reuse4": "facet4", "reuse_reject": "reject_only", "reuse_cosine": "cosine_only"}
CANDIDATES = 8
FOLDS = 4


class FacetIndex:
    def __init__(self, data, questions, encoder):
        with (data / "abstracts-csfcube-preds.jsonl").open() as stream:
            papers = {int(row["paper_id"]): row for row in map(json.loads, stream)}
        full_ids = sorted({doc for question in questions for doc in question["grades"]})
        facet_ids = sorted({(doc, question["facet"]) for question in questions for doc in question["grades"]})
        full = {doc: render(papers[doc]) for doc in full_ids}
        self.compact = {item: render(papers[item[0]], item[1]) for item in facet_ids}
        self.full_pos = {doc: i for i, doc in enumerate(full_ids)}
        self.full = encoder.encode(list(full.values()))
        unique = {text: i for i, text in enumerate(dict.fromkeys(self.compact.values()))}
        self.facet_pos = {item: unique[text] for item, text in self.compact.items()}
        self.facet = encoder.encode(list(unique))
        self.queries = {}
        for question in questions:
            doc, facet = question["query_id"], question["facet"]
            template = f"Find a paper with a similar {FACETS[facet]}.\n"
            self.queries[doc, facet] = (template + render(papers[doc]), template + render(papers[doc], facet))

    def nominate(self, question, encoder, view):
        qid, facet = question["query_id"], question["facet"]
        primary, focal = self.queries[qid, facet]
        vector = encoder.encode([primary], query=True)[0]
        universe = sorted(question["grades"])
        scores = self.full[[self.full_pos[doc] for doc in universe]] @ vector
        chosen = [universe[i] for i in np.argsort(-scores, kind="stable")[:CANDIDATES]]
        if len(chosen) != CANDIDATES:
            raise ValueError("Insufficient judged papers")
        cosine = {}
        if view != "none":
            candidates = self.facet[[self.facet_pos[doc, facet] for doc in chosen]]
            if view in ("focus", "both"):
                focused_vector = encoder.encode([focal], query=True)[0]
                cosine["focus"] = candidates @ focused_vector
            if view in ("reuse", "both"):
                cosine["reuse"] = candidates @ vector
        return focal, chosen, cosine

    def check(self, reader, text, documents, facet):
        return margins(reader, [(text, self.compact[doc, facet]) for doc in documents], TASK)

    def size(self):
        return dict(full_vectors=len(self.full), facet_vectors=len(self.facet),
                    index_vector_bytes=self.full.nbytes + self.facet.nbytes)


def collect(questions, index, encoder, reader):
    samples = []
    for question in questions:
        focal, docs, cosine = index.nominate(question, encoder, "both")
        values, clipped = index.check(reader, focal, docs, question["facet"])
        grades = np.asarray([question["grades"][doc] for doc in docs])
        samples.append(dict(docs=docs, **cosine, values=values, grades=grades, clipped=clipped))
    return samples


def fit_split(fit, view):
    grades = np.concatenate([row["grades"] for row in fit])
    logits = np.concatenate([row["values"] for row in fit])
    cosine = np.concatenate([row[view] for row in fit])
    reader = operating_point(logits, grades)
    positive = np.sort(cosine[(grades >= 2) & (logits > reader)])
    if len(positive) == 0:
        raise ValueError("Cannot calibrate the negative gate with no recognized positive")
    low = float(positive[len(positive) // 20])
    negative = np.sort(cosine[grades < 2])[::-1]
    high = max(low, float(negative[len(negative) // 50]))
    return dict(reader=reader, low=low, high=high, cosine=operating_point(cosine, grades),
                fit_pairs=len(grades), fit_verified_strong=len(positive))


def assignments(questions, samples):
    query_ids = {question["query_id"] for question in questions}
    groups = {query: i % FOLDS for i, query in enumerate(sorted(query_ids, key=key))}
    folds = [groups[question["query_id"]] for question in questions]
    policies = {}
    for fold in range(FOLDS):
        # Exclude exact source IDs appearing as query or selected evidence in the test fold.
        unavailable = set()
        for question, sample, assigned in zip(questions, samples, folds, strict=True):
            if assigned == fold:
                unavailable.update([question["query_id"], *sample["docs"]])
        retained = []
        for question, sample, assigned in zip(questions, samples, folds, strict=True):
            if assigned == fold or question["query_id"] in unavailable:
                continue
            keep = [i for i, doc in enumerate(sample["docs"]) if doc not in unavailable]
            if keep:
                retained.append({name: sample[name][keep] for name in ("grades", "values", "focus", "reuse")})
        policies[fold] = {view: fit_split(retained, view) for view in ("focus", "reuse")}
    return folds, policies


def act(arm, cosine, policy):
    check = np.ones(CANDIDATES, dtype=bool)
    automatic = np.zeros(CANDIDATES, dtype=bool)
    if arm == "dense4":
        check[4:] = False
    elif arm == "facet4":
        check[:] = False
        check[np.argsort(-cosine, kind="stable")[:4]] = True
    elif arm in ("reject_only", "two_sided"):
        check = cosine >= policy["low"]
        if arm == "two_sided":
            automatic = cosine > policy["high"]
            check &= ~automatic
    elif arm == "cosine_only":
        check[:] = False
        automatic = cosine > policy["cosine"]
    return np.flatnonzero(check), automatic


def returned(accepted):
    return np.flatnonzero(accepted)[:2].tolist()


def evaluate(questions, samples, index, encoder, reader, offset):
    folds, policies = assignments(questions, samples)
    rows = defaultdict(list)
    for i in range(-1, len(questions)):
        j = max(i, 0)
        question = questions[j]
        sample = samples[j]
        shift = (j + offset) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            view = "reuse" if arm in REUSE else "focus"
            policy = policies[folds[j]][view]
            method = REUSE.get(arm, arm)
            focal, docs, cosine = index.nominate(
                question, encoder, "none" if arm in ("dense8", "dense4") else view,
            )
            checked, accepted = act(method, cosine.get(view), policy)
            values, clipped = index.check(reader, focal, [docs[k] for k in checked], question["facet"])
            accepted[checked] = values > policy["reader"]
            picks = returned(accepted)
            json.dumps({"sources": [docs[j] for j in picks]})
            elapsed = (perf_counter() - start) * 1000
            if docs != sample["docs"]:
                raise ValueError("An arm changed the fixed full shortlist")
            if i >= 0:
                static, replay_mask = act(method, sample[view], policy)
                replay_mask[static] = sample["values"][static] > policy["reader"]
                rows[arm].append(dict(
                    grades=sample["grades"][picks].tolist(), checks=len(checked), clipped=clipped,
                    different_from_shared_replay=picks != returned(replay_mask), ms=elapsed,
                ))
    summary = {}
    groups = [q["query_id"] for q in questions]
    for arm, values in rows.items():
        strongest = [any(grade >= 2 for grade in row["grades"]) for row in values]
        base = [any(grade >= 2 for grade in row["grades"]) for row in rows["dense8"]]
        counts = Counter(grade for row in values for grade in row["grades"])
        summary[arm] = dict(
            issued_by_grade=[counts[grade] for grade in range(4)], strong_queries=sum(strongest),
            issued_queries=sum(bool(row["grades"]) for row in values),
            checks=sum(row["checks"] for row in values), clipped=sum(row["clipped"] for row in values),
            route_disagreements=sum(row["different_from_shared_replay"] for row in values),
            p50_p95_ms=np.quantile([row["ms"] for row in values], [.50, .95]).tolist(),
            gains=sum(a and not b for a, b in zip(strongest, base, strict=True)),
            losses=sum(b and not a for a, b in zip(strongest, base, strict=True)),
            paired_delta=clustered_paired_interval([int(a) - int(b) for a, b in zip(strongest, base, strict=True)], groups),
        )
    return dict(policies=policies, arms=summary, queries=len(questions), query_articles=len(set(groups)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "provenance", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    _, questions, sources = load(args.data)
    receipt = json.loads(args.provenance.read_bytes())
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    if (sources != receipt["source_hashes"] or provenance["revision"] != receipt["embedding"]["revision"]
        or reader.provenance["revision"] != receipt["reranker"]["revision"]):
        raise ValueError("Unexpected frozen dataset or model revision")
    start = perf_counter()
    index = FacetIndex(args.data, questions, encoder)
    offline_ms = (perf_counter() - start) * 1000
    samples = collect(questions, index, encoder, reader)
    result = evaluate(questions, samples, index, encoder, reader, args.order_offset)
    write_json(args.output, dict(
        **result, **index.size(), index_build_ms=offline_ms,
        gpu=torch.cuda.get_device_name(), source_hashes=sources, order_offset=args.order_offset,
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
