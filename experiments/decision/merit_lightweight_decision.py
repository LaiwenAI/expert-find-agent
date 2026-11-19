"""Measure a small suitability head with paper- and profile-disjoint MERIT splits."""

import argparse
from collections import defaultdict
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.decision.merit_evidence_dispatch import Decision, read_cases, terms
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint

SALT = "merit-profile-disjoint-v1"
ARMS = ("reader_two", "reader_full", "lexical_head", "dense_max", "dense_head", "dense_gate")
TASK = "Retrieve a publication that shows relevant technical expertise for reviewing the target paper."
MAX_ESCALATION = 0.2


def normalize(text):
    return " ".join(text.casefold().split())


def split_cases(cases):
    parent = list(range(len(cases)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    signatures = []
    for i, case in enumerate(cases):
        title_signature = tuple(sorted(normalize(p.split("\n", 1)[0]) for p in case.papers))
        signatures.append(title_signature)
        for key in (("target", case.cluster), ("profile", title_signature)):
            other = seen.setdefault(key, i)
            a, b = root(i), root(other)
            parent[max(a, b)] = min(a, b)
    components = defaultdict(list)
    for i in range(len(cases)):
        components[root(i)].append(i)
    output = dict(fit=[], calibration=[], test=[])
    for indices in components.values():
        component = sha256(f"{SALT}:{min(indices)}".encode()).hexdigest()
        bucket = int(component[:8], 16) % 10
        key = "fit" if bucket < 5 else "calibration" if bucket < 7 else "test"
        output[key].extend((component, cases[i]) for i in indices)
    for values in ([case.cluster for case in cases], signatures):
        assignments = defaultdict(set)
        for i, key in enumerate(values):
            assignments[key].add(root(i))
        if any(len(components) != 1 for components in assignments.values()):
            raise AssertionError("Split failed to group an exact target/profile signature")
    return output


def summarize_similarities(values):
    desc = np.sort(np.asarray(values))[::-1]
    return [float(desc[0]), float(desc[1]), float(desc[:3].mean()), float(desc.mean()),
            float(desc[0] - desc[1]), float(desc.std())]


class Representations:
    def __init__(self, cases, checkpoint):
        provenance = verify_checkpoint(checkpoint)
        self.embedding = QwenTorchEmbedding(checkpoint)
        self.provenance = provenance
        self.embedding.model.config.use_cache = False
        texts = sorted({p for case in cases for p in case.papers})
        self.vectors = dict(zip(texts, self.embedding.encode(texts), strict=True))
        self.terms = {p: terms(p) for p in texts}

    def lexical(self, case):
        target = terms(case.query)
        denom = max(1, len(target))
        rows = [self.terms[p] for p in case.papers]
        overlaps = [row & target for row in rows]
        similarities = [len(x) / max(1, len(target | row)) for x, row in zip(overlaps, rows, strict=True)]
        return summarize_similarities(similarities) + [
            len(set.union(set(), *overlaps)) / denom, len(case.papers), len(target)]

    def dense(self, case):
        encoded = self.embedding.encode([f"Instruct: {TASK}\nQuery: {case.query}"])[0]
        values = np.stack([self.vectors[p] for p in case.papers]) @ encoded
        return summarize_similarities(values)


def fit(features, labels):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    matrix = np.column_stack([np.ones(len(features)), (features-center)/scale])
    weights = np.zeros(matrix.shape[1])
    penalty = np.diag([0.] + [8.] * (matrix.shape[1]-1))
    for _ in range(20):
        linear = np.clip(matrix @ weights, -35, 35)
        p = 1 / (1 + np.exp(-linear))
        weights -= np.linalg.solve(
            matrix.T @ ((p*(1-p))[:, None] * matrix) + penalty,
            matrix.T @ (p-labels) + penalty @ weights,
        )
    return center, scale, weights


def logit(features, model):
    center, scale, weights = model
    return float(weights[0] + ((np.asarray(features)-center)/scale) @ weights[1:])


def balanced(predictions, gold):
    prediction = np.asarray(predictions, dtype=bool)
    label = np.asarray(gold, dtype=bool)
    return float((np.mean(prediction[label]) + np.mean(~prediction[~label])) / 2)


def threshold(scores, gold):
    a = np.asarray(scores)
    choices = sorted(set(a) | {max(a)+1., min(a)-1.})
    return float(max(choices, key=lambda t: (balanced(a >= t, gold), -abs(t))))


def calibrate_gate(scores, slow, gold, fast_threshold, slow_threshold):
    fast = np.asarray(scores)
    base = fast >= fast_threshold
    margins = np.abs(fast-fast_threshold)
    replacements = np.asarray(slow) >= slow_threshold
    choices = []
    for cutoff in sorted(set(margins) | {-1.}):
        used = margins <= cutoff
        if np.mean(used) <= MAX_ESCALATION:
            choices.append((balanced(np.where(used,replacements,base),gold), -int(used.sum()), cutoff))
    return float(max(choices)[-1])


def measure(arm, case, index, reader, models, cutoffs, gate):
    start = perf_counter()
    escalated = False
    if arm in ("reader_two", "reader_full"):
        score, _ = reader.judge(case.target, case.papers if arm == "reader_full" else case.papers[:2])
        predicted = score >= cutoffs[arm]
    else:
        if arm == "lexical_head":
            score = logit(index.lexical(case), models[arm])
        elif arm == "dense_max":
            score = index.dense(case)[0]
        else:
            score = logit(index.dense(case) + index.lexical(case), models["dense_head"])
        target = "dense_head" if arm == "dense_gate" else arm
        predicted = score >= cutoffs[target]
        if arm == "dense_gate" and abs(score-cutoffs[target]) <= gate:
            escalated = True
            score, _ = reader.judge(case.target, case.papers[:2])
            predicted = score >= cutoffs["reader_two"]
    return dict(prediction=int(predicted), score=score, escalated=escalated, ms=1000*(perf_counter()-start))


def summarize(test):
    labels = np.array([case.label for _,case,_ in test], dtype=bool)
    groups = [group for group,_,_ in test]
    baseline = np.array([row["reader_two"]["prediction"] for _,_,row in test], dtype=bool) == labels
    result = {}
    for arm in ARMS:
        values = [row[arm] for _,_,row in test]
        predictions = np.array([v["prediction"] for v in values], dtype=bool)
        correct = predictions == labels
        tp = int((predictions & labels).sum())
        result[arm] = dict(
            accuracy=float(correct.mean()), balanced_accuracy=balanced(predictions, labels),
            positive_f1=2*tp/max(1,int(predictions.sum())+int(labels.sum())), true_positives=tp,
            predicted_positives=int(predictions.sum()),
            escalated=sum(v["escalated"] for v in values), mean_ms=mean(v["ms"] for v in values),
            p50_p95_ms=np.quantile([v["ms"] for v in values],[.5,.95]).tolist(),
            accuracy_delta_vs_reader_two=clustered_paired_interval(
                correct.astype(int)-baseline.astype(int), groups),
        )
    return result


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    cases = read_cases(args.data)
    splits = split_cases(cases)
    start = perf_counter()
    index = Representations(cases, args.embedding)
    offline_seconds = perf_counter()-start
    reader = Decision(args.reader)
    models = {}
    fit_cases = [c for _,c in splits["fit"]]
    lexical = [index.lexical(c) for c in fit_cases]
    dense = [index.dense(c) for c in fit_cases]
    train_labels = np.array([c.label for c in fit_cases])
    models["lexical_head"] = fit(np.array(lexical), train_labels)
    models["dense_head"] = fit(np.array([d+l for d,l in zip(dense,lexical,strict=True)]),train_labels)
    calibration = []
    for _,case in splits["calibration"]:
        l, d = index.lexical(case), index.dense(case)
        small, _ = reader.judge(case.target,case.papers[:2])
        full, _ = reader.judge(case.target,case.papers)
        calibration.append((case.label, {"lexical_head":logit(l,models["lexical_head"]), "dense_max":d[0],
             "dense_head":logit(d+l,models["dense_head"]), "reader_two":small, "reader_full":full}))
    gold = [label for label,_ in calibration]
    scores = {arm:[row[arm] for _,row in calibration] for arm in ARMS if arm!="dense_gate"}
    cutoffs = {arm:threshold(value,gold) for arm,value in scores.items()}
    gate = calibrate_gate(scores["dense_head"], scores["reader_two"], gold,
                          cutoffs["dense_head"], cutoffs["reader_two"])
    print("Completed fit and calibration; splits:", {k:len(v) for k,v in splits.items()}, flush=True)
    report = []
    for i,(group,case) in enumerate(splits["test"]):
        offset = i%len(ARMS)
        report.append((group,case,{arm:measure(arm,case,index,reader,models,cutoffs,gate)
                                   for arm in ARMS[offset:]+ARMS[:offset]}))
        if (i+1)%30==0:
            print(f"Completed test: {i+1}",flush=True)
    write_json(args.output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        data_sha256=digest(args.data), script_sha256=digest(Path(__file__)), embeddings=index.provenance["revision"],
        reader=reader.provenance["revision"], offline_seconds=offline_seconds, source_vectors=len(index.vectors),
        group_guard="connected components of target title or exact normalized profile title multiset",
        splits={name:dict(pairs=len(members),positives=sum(case.label for _,case in members),
            components=len({group for group,_ in members})) for name,members in splits.items()},
        cutoffs=cutoffs, gate=gate, maximum_calibration_escalation=MAX_ESCALATION, arms=summarize(report)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","embedding","reader","output"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
