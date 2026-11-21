"""Fit resident role heads on synthetic matched task contrasts."""

from hashlib import sha256
import json

import numpy as np
import torch
from torch.nn import functional as F

from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_role_banks import encode
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest

ARMS = ("original_ce", "augmented_ce", "shuffled_pair_ce", "matched_pair_ce")
STEPS = 120
RATE = .002


def training_examples(path, original, wording):
    bank = json.loads(path.read_bytes())
    invalid = {normalize(text) for text in original}
    invalid.update(normalize(text) for view in wording.values() for _, _, text in view)
    seen_roles, added, labels, matched, shuffled = [], [], [], [], []
    for entry in bank:
        a, b = map(ROLES.index, entry["roles"])
        seen_roles.extend((a, b))
        start = len(original) + len(added)
        pairs = entry["pairs"]
        if len(pairs) != 4:
            raise ValueError("Expected four same-topic contrasts per role edge")
        for index, (left, right) in enumerate(pairs):
            for text in (left, right):
                normalized = normalize(text)
                if not normalized or normalized in invalid:
                    raise ValueError("Repeated training or evaluation wording")
                invalid.add(normalized)
                added.append(text)
            labels.extend((a, b))
            left_index = start + 2 * index
            matched.append((left_index, left_index + 1, a, b))
            other = start + 2 * ((index + 1) % len(pairs)) + 1
            shuffled.append((left_index, other, a, b))
    if sorted(seen_roles) != list(range(len(ROLES))):
        raise ValueError("Expected one matched role edge for each role")
    return added, labels, matched, shuffled


def fit(vectors, labels, contrasts=()):
    targets = F.one_hot(labels, len(ROLES)).to(vectors.dtype)
    kernel = vectors @ vectors.T
    comparison = kernel + torch.eye(len(labels), device=vectors.device) * .1
    initial = torch.linalg.solve(comparison, targets).T @ vectors
    weights = torch.nn.Parameter(initial * 10)
    optimizer = torch.optim.AdamW((weights,), lr=RATE)
    if contrasts:
        left, right, role, rival = torch.as_tensor(contrasts, device=vectors.device).T

    for _ in range(STEPS):
        optimizer.zero_grad()
        logits = vectors @ weights.T
        loss = F.cross_entropy(logits, labels)
        if contrasts:
            left_margin = logits[left, role] - logits[left, rival]
            right_margin = logits[right, role] - logits[right, rival]
            loss = loss + .5 * F.softplus(-(left_margin - right_margin)).mean()
        loss.backward()
        optimizer.step()
    return weights.detach().cpu().numpy()


class RoleContrastHeads:
    def __init__(self, path, encoder, original, wording):
        added, labels, matched, shuffled = training_examples(path, original, wording)
        rows = torch.from_numpy(encode(encoder, original + added)).to("cuda")
        base = torch.arange(len(ROLES), device=rows.device).repeat_interleave(6)
        expanded = torch.cat((base, torch.as_tensor(labels, device=rows.device)))
        size = len(original)
        self.weights = {
            "original_ce": fit(rows[:size], base),
            "augmented_ce": fit(rows, expanded),
            "shuffled_pair_ce": fit(rows, expanded, shuffled),
            "matched_pair_ce": fit(rows, expanded, matched),
        }
        self.provenance = {
            "training_sha256": digest(path),
            "original_rows": size,
            "added_rows": len(added),
            "matched_pairs": len(matched),
            "cuda_steps": STEPS,
            "learning_rate": RATE,
            "weight_sha256": {arm: sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()
                              for arm, weights in self.weights.items()},
        }

    def predict(self, features):
        return {arm: [ROLES[i] for i in (features @ weights.T).argmax(axis=1)]
                for arm, weights in self.weights.items()}
