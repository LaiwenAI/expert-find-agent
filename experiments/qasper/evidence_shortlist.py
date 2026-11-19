"""Compare evidence complementarity under a fixed QASPER reranking budget."""

import argparse
from collections import Counter
from hashlib import sha256
import math
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.qasper import adaptive_reranking_depth as depth
from experiments.qasper import certified_reranking as base
from experiments.qasper.adaptive_evidence import STOP


BUDGET = 16
ANCHORS = 8
POOL = 32
WEIGHTS = (.10, .25, .50, .75)
POLICIES = ("dense", "sections", "mmr", "coverage")
REPLICATION_SIZE = 144
REPLICATION_SALT = "qasper-evidence-shortlist-replication-v1"
WORD = re.compile(r"[a-z][a-z0-9]+")


def terms(text):
    return [word for word in WORD.findall(text.lower()) if word not in STOP]


class CandidatePool:
    def __init__(self, question, scores, vectors, statistics, sections):
        self.order = np.argsort(-scores, kind="stable")[:POOL]
        self.sections = [sections[index] for index in self.order]
        ordered = scores[self.order]
        spread = max(1e-12, ordered[0] - ordered[-1])
        self.relevance = (ordered - ordered[-1]) / spread
        self.vectors = vectors[self.order]
        self.query_terms = sorted(set(terms(question)))
        self.statistics = statistics

    def coverage(self):
        counts, document_frequency = self.statistics
        counts_in_pool = np.array([[counts[index][word] for word in self.query_terms]
                                   for index in self.order], dtype=float)
        # Saturating evidence across passages discounts repeated mentions.
        contribution = counts_in_pool / (counts_in_pool + 1)
        inverse_frequency = np.array([
            math.log1p((len(counts) - document_frequency[word]) / (1 + document_frequency[word]))
            for word in self.query_terms
        ])
        return contribution, inverse_frequency

    def choose(self, policy, weight=0.0):
        if policy == "fixed32":
            return self.order.tolist()
        count = min(BUDGET, len(self.order))
        if policy == "dense" or count <= ANCHORS:
            return self.order[:count].tolist()
        selected = list(range(ANCHORS))
        available = np.ones(len(self.order), dtype=bool)
        available[:ANCHORS] = False
        seen = set(self.sections[:ANCHORS])
        if policy == "coverage":
            contribution, inverse_frequency = self.coverage()
            remaining = inverse_frequency * np.prod(1 - contribution[:ANCHORS], axis=0)
        if policy == "mmr":
            similarity = np.maximum(self.vectors @ self.vectors.T, 0)
        for _ in range(count - ANCHORS):
            if policy == "sections":
                fresh = available & np.array([section not in seen for section in self.sections])
                scores = np.where(fresh.any(), fresh, available)
            elif policy == "mmr":
                scores = self.relevance - weight * similarity[:, selected].max(axis=1)
            elif policy == "coverage":
                gain = contribution @ remaining
                scale = gain[available].max()
                scores = self.relevance + weight * gain / max(scale, 1e-12)
            else:
                raise ValueError(f"Unknown shortlist policy: {policy}")
            chosen = int(np.argmax(np.where(available, scores, -np.inf)))
            selected.append(chosen)
            available[chosen] = False
            seen.add(self.sections[chosen])
            if policy == "coverage":
                remaining *= 1 - contribution[chosen]
        # Present selected candidates in dense order for all policies.
        return self.order[sorted(selected)].tolist()


def paper_indexes(dataset, selection, encoder):
    meta, _, vectors, offline = base.offline({pid: dataset[pid] for pid, _ in selection}, encoder)
    start = perf_counter()
    indexes = {}
    for pid, (paragraphs, sections, lo, hi) in meta.items():
        counts = [Counter(terms(text)) for text in paragraphs]
        frequencies = Counter(word for count in counts for word in count)
        indexes[pid] = (paragraphs, sections, np.ascontiguousarray(vectors[lo:hi], dtype=np.float64),
                        (counts, frequencies))
    return indexes, dict(**offline, lexical_index_seconds=perf_counter()-start)


def rerank(reader, question, context, selected, dense):
    scored = []
    truncated = 0
    for start in range(0, len(selected), base.BLOCK):
        indices = selected[start:start+base.BLOCK]
        margins, clipped = reader.score(question, context, indices, task=base.TASK)
        if not all(math.isfinite(margin) for margin in margins):
            raise ValueError("Nonfinite reranker margin")
        for index, margin in zip(indices, margins, strict=True):
            probability = 1 / (1 + math.exp(-max(-40, min(40, margin))))
            fused = (1 - base.WEIGHT) * dense[index] + base.WEIGHT * probability
            scored.append((-float(fused), index))
        truncated += clipped
    return [index for _, index in sorted(scored)[:base.TOP]], truncated


def candidate_quality(paragraphs, shortlist, gold):
    results = base.quality([paragraphs[index] for index in shortlist], gold)
    return None if results is None else results["complete_evidence"]


def online_scores(encoder, question, vectors):
    return vectors @ np.asarray(encoder.encode([base.query_input(question)])[0], dtype=np.float64)


def calibrate(encoder, selection, indexes):
    trials = {policy: {weight: [] for weight in WEIGHTS} for policy in ("mmr", "coverage")}
    for pid, qa in selection:
        paragraphs, sections, vectors, statistics = indexes[pid]
        gold = base.evidence_sets(qa, paragraphs)
        if not gold:
            continue
        question = qa["question"]
        dense = online_scores(encoder, question, vectors)
        pool = CandidatePool(question, dense, vectors, statistics, sections)
        for policy, values in trials.items():
            for weight, observations in values.items():
                shortlist = pool.choose(policy, weight)
                observations.append(candidate_quality(paragraphs, shortlist, gold))
    return {
        policy: dict(selected=max(values, key=lambda weight: (mean(values[weight]), -weight)),
                     grid={str(weight): mean(observations) for weight, observations in values.items()},
                     evaluable=len(values[WEIGHTS[0]]))
        for policy, values in trials.items()
    }


def replication_panel(dataset):
    exclude = {pid for pid, _ in base.panel(dataset)}

    def order(identity):
        return sha256(f"{REPLICATION_SALT}:{identity}".encode()).digest()

    papers = sorted(dataset.keys() - exclude, key=order)[:REPLICATION_SIZE]
    return [(pid, min(dataset[pid]["qas"], key=lambda qa: order(qa["question_id"]))) for pid in papers]


def evaluate(encoder, reader, selection, indexes, calibration, policies):
    rows = []
    for i, (pid, qa) in enumerate(selection):
        paragraphs, sections, vectors, statistics = indexes[pid]
        question = qa["question"]
        context = [f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)]
        gold = base.evidence_sets(qa, paragraphs)
        offset = i % len(policies)
        arms = {}
        for policy in policies[offset:] + policies[:offset]:
            start = perf_counter()
            dense = online_scores(encoder, question, vectors)
            pool = CandidatePool(question, dense, vectors, statistics, sections)
            shortlist = pool.choose(policy, calibration.get(policy, {}).get("selected", 0.0))
            order, truncated = rerank(reader, question, context, shortlist, dense)
            elapsed_ms = (perf_counter()-start) * 1000
            limit = POOL if policy == "fixed32" else BUDGET
            if len(set(shortlist)) != len(shortlist) or len(shortlist) != min(len(paragraphs), limit):
                raise ValueError("A shortlist violated the fixed candidate budget")
            arms[policy] = dict(shortlist=shortlist, order=order, truncated=truncated, ms=elapsed_ms,
                                shortlist_complete=candidate_quality(paragraphs, shortlist, gold),
                                evidence=base.quality([paragraphs[index] for index in order], gold))
        rows.append(dict(paper_id=pid, question_id=qa["question_id"], arms=arms))
        if len(rows) % 16 == 0:
            print("Completed matched requests:", len(rows), flush=True)
    return rows


def summarize(rows, policies):
    scored = [row for row in rows if row["arms"]["dense"]["evidence"] is not None]
    result = {}
    for policy in policies:
        observations = [row["arms"][policy] for row in rows]
        labeled = [row["arms"][policy] for row in scored]
        paired = [row["arms"]["dense"] for row in scored]
        result[policy] = dict(
            p95_ms=float(np.quantile([row["ms"] for row in observations], .95)),
            mean_rerank_checks=mean(len(row["shortlist"]) for row in observations),
            shortlist_complete=mean(row["shortlist_complete"] for row in labeled),
            returned_complete=mean(row["evidence"]["complete_evidence"] for row in labeled),
            evidence_f1=mean(row["evidence"]["f1"] for row in labeled),
            shortlist_delta=clustered_paired_interval([
                a["shortlist_complete"] - b["shortlist_complete"]
                for a, b in zip(labeled, paired, strict=True)
            ], [row["paper_id"] for row in scored]),
            returned_delta=clustered_paired_interval([
                a["evidence"]["complete_evidence"] - b["evidence"]["complete_evidence"]
                for a, b in zip(labeled, paired, strict=True)
            ], [row["paper_id"] for row in scored]),
        )
    return dict(requests=len(rows), evaluable=len(scored), policies=result)


def run(data, embedding, reranker, output, replication=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output path and a Slurm allocation")
    development = base.read_data(data, "dev")
    test = base.read_data(data, "test")
    if development.keys() & test.keys():
        raise ValueError("Development and test paper IDs overlap")
    calibration_selection = [(pid, qa) for pid, qa in depth.selected_dev(development)
                             if depth.partition(pid) == "calibration"]
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
    dev_indexes, dev_offline = paper_indexes(development, calibration_selection, encoder)
    weights = calibrate(encoder, calibration_selection, dev_indexes)
    selection = replication_panel(test) if replication else base.panel(test)
    policies = (*POLICIES, "fixed32") if replication else POLICIES
    test_indexes, test_offline = paper_indexes(test, selection, encoder)
    rows = evaluate(encoder, reader, selection, test_indexes, weights, policies)
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=encoder.torch.cuda.get_device_name(), data_assets=base.ASSETS,
        embedding=embedding_info, reranker=reader.provenance, script_sha256=base.digest(Path(__file__)),
        ranker_script_sha256=base.digest(Path(base.__file__)), depth_script_sha256=base.digest(Path(depth.__file__)),
        config=dict(budget=BUDGET, anchors=ANCHORS, pool=POOL, weights=WEIGHTS,
                    sample_salt=REPLICATION_SALT if replication else base.SALT),
        offline=dict(development=dev_offline, test=test_offline), calibration=weights,
        report=summarize(rows, policies), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data", "embedding", "reranker", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--replication", action="store_true")
    run(**vars(parser.parse_args()))
