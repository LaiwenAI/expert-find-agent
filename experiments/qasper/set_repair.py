"""Fit a set-level repair policy from annotated complete evidence sets."""

import argparse
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
import json
import os
from pathlib import Path
from statistics import mean
import tarfile
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.qasper import certified_reranking as base
from experiments.qasper.evidence_gap_audit import choose, score
from experiments.qasper.evidence_shortlist import online_scores, paper_indexes, terms


TRAIN_SHA = "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"
ARMS = ("pointwise", "additive", "joint_one", "joint_two")
THRESHOLDS = (0., .05, .10, .20, .40, .80, None)
ADDITIVE_FEATURES = 6
POOL = 16
WIDTH = base.TOP


def read_train(path):
    with tarfile.open(path, "r:gz") as archive:
        stream = archive.extractfile("qasper-train-v0.3.json")
        if stream is None:
            raise ValueError("Missing QASPER training data")
        raw = stream.read()
    if sha256(raw).hexdigest() != TRAIN_SHA:
        raise ValueError("Changed QASPER training data")
    return json.loads(raw)


@lru_cache
def alternatives(n, width):
    """Enumerate fixed-size sets within two swaps of the pointwise set."""
    primary = tuple(range(width))
    extra = tuple(range(width, n))
    sets = [primary]
    for k in (1, 2):
        for removed, added in product(combinations(primary, k), combinations(extra, k)):
            sets.append(tuple(i for i in primary if i not in removed) + added)
    return np.array(sets, dtype=np.int64), 1 + len(primary) * len(extra)


def features(question, ordered, fused, dense, sections, vectors, stats):
    candidates, one_swap = alternatives(len(ordered), min(WIDTH, len(ordered)))
    rank = np.asarray(ordered)
    query_words = sorted(set(terms(question)))
    counts, frequency = stats
    idf = np.array([1 / (1 + frequency[word]) for word in query_words])
    mentions = np.array([[word in counts[i] for word in query_words] for i in ordered], dtype=bool)
    hits = mentions[candidates].sum(axis=1)
    coverage = (hits > 0) @ idf / max(idf.sum(), 1e-12)
    overlap = np.maximum(hits - 1, 0) @ idf / max(idf.sum(), 1e-12)
    source_similarity = vectors[rank] @ vectors[rank].T
    source_sections = np.asarray(sections)[rank]
    left, right = np.triu_indices(candidates.shape[1], 1)
    if len(left):
        a, b = candidates[:, left], candidates[:, right]
        similarities = source_similarity[a, b]
        same_section = source_sections[a] == source_sections[b]
        distance = np.abs(rank[a] - rank[b])
        interaction = [similarities.mean(axis=1), similarities.max(axis=1), similarities.min(axis=1),
                       same_section.mean(axis=1), ((distance <= 2) & same_section).mean(axis=1),
                       np.log1p(distance).mean(axis=1)]
    else:
        interaction = [np.zeros(len(candidates)) for _ in range(6)]
    chosen = candidates
    view = np.column_stack([
        fused[chosen].mean(axis=1), fused[chosen].min(axis=1), fused[chosen].std(axis=1),
        dense[rank[chosen]].mean(axis=1), dense[rank[chosen]].min(axis=1), chosen.mean(axis=1) / POOL,
        coverage, overlap, *interaction,
    ])
    return candidates, view - view[0], one_swap


def gold_sets(paragraphs, ordered, candidate_sets, refs):
    positions = {paragraphs[i]: j for j, i in enumerate(ordered)}
    mask = np.zeros(len(candidate_sets), dtype=bool)
    for reference in refs:
        if len(reference) <= WIDTH and reference <= positions.keys():
            required = np.fromiter((positions[text] for text in reference), int)
            mask |= np.isin(candidate_sets, required).sum(axis=1) == len(required)
    return mask


def observe(dataset, panel, encoder, reader):
    indexes, offline = paper_indexes(dataset, panel, encoder)
    rows = []
    for i, (pid, qa) in enumerate(panel):
        question = qa["question"]
        paragraphs, sections, vectors, statistics = indexes[pid]
        begin = perf_counter()
        dense = online_scores(encoder, question, vectors)
        pool = np.argsort(-dense, kind="stable")[:POOL].tolist()
        context = [f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)]
        scored, _ = score(reader, question, context, dense, pool)
        scored.sort(key=lambda row: (-row[0], row[1]))
        fused, ordered = zip(*scored, strict=True)
        prefix_ms = (perf_counter()-begin)*1000
        begin = perf_counter()
        sets, feature, one_swap = features(question, ordered, np.asarray(fused), dense, sections, vectors, statistics)
        feature_ms = (perf_counter()-begin)*1000
        refs = base.evidence_sets(qa, paragraphs)
        labels = gold_sets(paragraphs, ordered, sets, refs)
        rows.append(dict(view=feature, labels=labels, one_swap=one_swap, prefix_ms=prefix_ms, feature_ms=feature_ms,
                         reachable=any(ref <= {paragraphs[j] for j in pool} for ref in refs)))
        if (i+1) % 60 == 0:
            print("Prepared questions:", i+1, flush=True)
    return rows, offline


def fit(rows, columns):
    rng = np.random.default_rng(20261119)
    deltas = []
    for row in rows:
        positive = np.flatnonzero(row["labels"])
        negative = np.flatnonzero(~row["labels"])
        if len(positive) and len(negative):
            # Pair each reachable request equally; inactive requests cannot supply a preference.
            pos_ids = rng.choice(positive, 64, replace=True)
            neg_ids = rng.choice(negative, 64, replace=True)
            differences = row["view"][pos_ids, :columns] - row["view"][neg_ids, :columns]
            deltas.append(differences)
    x = np.vstack(deltas)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
    design = x / scale
    weights = np.linalg.solve(design.T @ design + 32*np.eye(columns), design.sum(axis=0)) / scale
    return weights, len(deltas)


def preference(row, model, arm):
    if arm == "pointwise":
        return 0, 0.
    features = row["view"][:row["one_swap"]] if arm == "joint_one" else row["view"]
    values = features[:, :len(model)] @ model
    candidate = int(values.argmax())
    return candidate, float(values[candidate])


def outcome(row, candidate, gain, threshold):
    if threshold is None or gain <= threshold:
        candidate = 0
    return bool(row["labels"][candidate]), candidate != 0


def calibrate(rows, models):
    result = {}
    for arm, model in models.items():
        best = None
        preferences = [preference(row, model, arm) for row in rows]
        for threshold in THRESHOLDS:
            output = [outcome(row, *selected, threshold) for row, selected in zip(rows, preferences, strict=True)]
            complete = sum(success for success, _ in output)
            changes = sum(edited for _, edited in output)
            if best is None or (complete, -changes) > (best[0], -best[1]):
                best = (complete, changes, threshold)
        result[arm] = dict(complete=best[0], edited=best[1], threshold=best[2])
    return result


def evaluate(rows, models, calibration):
    arms = {}
    baseline = [bool(row["labels"][0]) for row in rows]
    for arm in (*ARMS, "forced_joint_two"):
        method = "joint_two" if arm.startswith("forced") else arm
        threshold = 0. if arm.startswith("forced") else calibration.get(arm, {"threshold": None})["threshold"]
        measures = []
        for row in rows:
            begin = perf_counter()
            candidate, gain = (0,0.) if threshold is None else preference(row, models[method], method)
            selected, changed = outcome(row, candidate, gain, threshold)
            extra = (perf_counter()-begin)*1000 + (row["feature_ms"] if threshold is not None else 0.)
            measures.append((selected, changed, row["prefix_ms"]+extra, extra))
        delta = [int(result[0])-int(ref) for result,ref in zip(measures, baseline, strict=True)]
        arms[arm] = dict(
            complete=sum(row[0] for row in measures), edited=sum(row[1] for row in measures),
            gains=delta.count(1), losses=delta.count(-1), paired_delta=paired_interval(delta),
            p95_ms=float(np.quantile([row[2] for row in measures], .95)),
            selection_p95_ms=float(np.quantile([row[3] for row in measures], .95)),
        )
    return arms


def run(data, archive, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    partitions = {"train":read_train(archive), "dev":base.read_data(data,"dev"), "test":base.read_data(data,"test")}
    if sum(map(len,partitions.values())) != len(set().union(*partitions.values())):
        raise ValueError("Overlapping official paper splits")
    provenance = base.verify_checkpoint(embedding)
    encoder = base.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    rows, panels = {}, {}
    for split,data_split in partitions.items():
        panel = choose(data_split,multi=True)
        batch, offline = observe(data_split,panel,encoder,reader)
        rows[split] = batch
        panels[split] = dict(
            questions=len(panel), offline=offline, reachable=sum(row["reachable"] for row in batch),
            one_swap_ceiling=sum(int(row["labels"][:row["one_swap"]].any()) for row in batch),
            two_swap_ceiling=sum(int(row["labels"].any()) for row in batch),
            panel_sha256=sha256("\n".join(qa["question_id"] for _,qa in panel).encode()).hexdigest(),
        )
    additive, pairs = fit(rows["train"],ADDITIVE_FEATURES)
    joint, joint_pairs = fit(rows["train"],rows["train"][0]["view"].shape[1])
    models = dict(additive=additive,joint_one=joint,joint_two=joint)
    settings = calibrate(rows["dev"],models)
    base.write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),panels=panels,
        train_sha256=TRAIN_SHA,data_sha256=base.ASSETS,script_sha256=base.digest(Path(__file__)),
        embedding_revision=provenance["revision"],reranker_revision=reader.provenance["revision"],
        fit_comparable_questions=[pairs,joint_pairs],calibration=settings,
        results=evaluate(rows["test"],models,settings),
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","archive","embedding","reranker","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
