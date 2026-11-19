"""Learn a lightweight duty selector over the existing retrieval vector."""

import json
from time import perf_counter

import numpy as np

from experiments.credit import anchored_role_margin as roles
from experiments.credit import role_applicability_gate as gate
from experiments.followir.core17_reranking import digest


TRAINING_SHA256 = "3365bdd2588d8dfff12de71cf7663808c25689f831a1ba42288a8bb8b53bfdc2"
NAMES = ("embedding_gap", "supervised_ridge", "distilled_ridge")
REGULARIZATION = .1
ROLE_INDEX = {role: index for index, role in enumerate(roles.ROLES)}
NISO_TO_LEGACY = {name: legacy for name, (legacy, _) in gate.POSITIVE.items()}


class DutyStudent:
    def __init__(self, training_path, encoder, definitions, reader, teacher_score, held_text):
        if digest(training_path) != TRAINING_SHA256:
            raise ValueError("Changed synthetic role training data")
        bank = json.loads(training_path.read_bytes())
        rows = [(value, NISO_TO_LEGACY.get(name)) for name, values in bank.items() for value in values]
        calibration = [(value, role) for role, value in roles.PARAPHRASES.items()]
        calibration.extend((value, None) for value in roles.OUTSIDE)
        train_text = [text for text, _ in rows]
        cal_text = [text for text, _ in calibration]
        partitions = [{self.normalize(text) for text in values} for values in (train_text, cal_text, held_text)]
        if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training, calibration and held phrases overlap")

        start = perf_counter()
        vectors = encoder.encode([gate.query(text) for text in train_text])
        targets = np.array([[float(role == expected) for role in roles.ROLES] for _, expected in rows])
        targets = 2. * targets - 1.
        teacher = np.array([list(teacher_score(reader, text, roles.ROLES).values()) for text in train_text])
        values = np.concatenate((targets, np.clip(teacher, -12., 12.)), axis=1).astype(np.float32)
        weights = np.linalg.solve(
            vectors @ vectors.T + REGULARIZATION * np.eye(len(vectors), dtype=np.float32), values,
        ).T @ vectors
        self.weights = dict(supervised_ridge=weights[:len(roles.ROLES)], distilled_ridge=weights[len(roles.ROLES):])
        fitting_ms = (perf_counter() - start) * 1000

        cal_scores = {name: [] for name in NAMES}
        for phrase, expected in calibration:
            vector = encoder.encode([gate.query(phrase)])[0]
            cosine = definitions @ vector
            candidates = np.argsort(-cosine, kind="stable")[:2]
            for name in NAMES:
                chosen, score = self.verdict(name, candidates, cosine, vector)
                cal_scores[name].append((score, chosen == expected))
        self.cuts = {name: max(score for score, good in values if not good) for name, values in cal_scores.items()}
        self.metadata = dict(
            training_sha256=digest(training_path), training_examples=len(rows),
            positive_examples=int((targets == 1).any(axis=1).sum()),
            teacher_calls=len(rows), fitting_ms=fitting_ms, weight_bytes=int(weights.nbytes),
            calibration_cases=len(calibration), regularization=REGULARIZATION,
            calibration={name: dict(cutoff=self.cuts[name], correct_retained=sum(
                good and score > self.cuts[name] for score, good in values))
                for name, values in cal_scores.items()},
        )

    @staticmethod
    def normalize(text):
        return " ".join(text.casefold().split())

    def verdict(self, name, candidates, cosine, vector):
        margins = cosine[candidates] if name == "embedding_gap" else self.weights[name][candidates] @ vector
        best = int(margins.argmax())
        gap = float(margins[best] - margins[1 - best])
        return roles.ROLES[candidates[best]], gap

    def decide(self, name, shortlist, cosine, vector):
        indices = np.array([ROLE_INDEX[head] for head in shortlist])
        head, margin = self.verdict(name, indices, cosine, vector)
        return head, margin > self.cuts[name]
