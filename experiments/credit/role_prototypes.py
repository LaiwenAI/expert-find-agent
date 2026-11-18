"""Construct role proposal banks from synthetic atomic task descriptions."""

from collections import Counter
import json
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import TASK


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def encode_tasks(embedder, tasks):
    return embedder.encode([f"Instruct: {TASK}\nQuery: {task}" for task in tasks])


def train_banks(path, router, examples):
    data = json.loads(path.read_bytes())
    if set(data) != set(ROLES) or {len(rows) for rows in data.values()} != {6}:
        raise ValueError("Expected six independent synthetic tasks per role")
    training = [phrase for role in ROLES for phrase in data[role]]
    keys = {normalized(phrase) for phrase in training}
    if len(keys) != len(training) or keys.intersection(normalized(row["text"]) for row in examples):
        raise ValueError("Duplicate training phrase or exact NISO overlap")

    started = perf_counter()
    features = torch.from_numpy(encode_tasks(router.embedder, training)).cuda()
    embedding_seconds = perf_counter() - started
    started = perf_counter()
    prototypes = F.normalize(features.reshape(len(ROLES), 6, -1).mean(dim=1), dim=1)
    labels = torch.arange(len(ROLES), device="cuda").repeat_interleave(6)
    targets = F.one_hot(labels, num_classes=len(ROLES)).to(features.dtype)
    # Solve ridge regression in the small sample space instead of the embedding dimension.
    kernel = features @ features.T + .1 * torch.eye(len(features), device=features.device)
    ridge = targets.T @ torch.linalg.solve(kernel, features)
    banks = {"definitions": router.vectors, "prototypes": prototypes.cpu().numpy(), "ridge": ridge.cpu().numpy()}
    return banks, dict(examples=len(training), feature_seconds=embedding_seconds,
                       fit_seconds=perf_counter() - started, ridge_penalty=.1)


def evaluate_atomic(router, banks, examples):
    vectors = encode_tasks(router.embedder, [row["text"] for row in examples])
    output = {}
    for name, bank in banks.items():
        predictions = []
        for row, scores in zip(examples, vectors @ bank.T, strict=True):
            order = np.argsort(-scores, kind="stable").tolist()
            lexical = router.lexical_rank(row["text"])[0][:3]
            predicted = ROLES[order[0]]
            gold = ROLES.index(row["role"])
            predictions.append(dict(ordinal=row["ordinal"], role=row["role"], prediction=predicted,
                                    dense_top3=gold in order[:3], union_top3=gold in (set(order[:3]) | set(lexical))))
        counts = Counter(p["role"] for p in predictions if p["role"] == p["prediction"])
        output[name] = dict(
            correct=sum(counts.values()), by_role=dict(counts),
            dense_top3=sum(p["dense_top3"] for p in predictions),
            union_top3=sum(p["union_top3"] for p in predictions),
            predictions=predictions,
        )
    return output
