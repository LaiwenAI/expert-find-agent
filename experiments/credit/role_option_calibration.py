"""Separate position and role-name priors with content-free prompts."""

import numpy as np

from experiments.credit.paired_role_verification import keyed_order
from experiments.credit.plos_role_inventory import ROLES


ARMS = (
    "teacher_position_corrected",
    "teacher_role_corrected",
    "teacher_joint_corrected",
    "teacher_reversed_order",
    "teacher_order_average",
)
PROBES = ("N/A", "[MASK]", "")


def unchanged_order(_phrase, roles):
    return list(roles)


def reversed_order(phrase, roles):
    return keyed_order(phrase, roles)[::-1]


class RoleOptionCalibration:
    def __init__(self, reader):
        self.reader = reader
        rotations = [ROLES[i:] + ROLES[:i] for i in range(len(ROLES))]
        position_samples = []
        role_samples = {role: [] for role in ROLES}
        self.probe_tokens = 0
        for phrase in PROBES:
            for start in range(0, len(rotations), 7):
                batch = rotations[start:start + 7]
                prompts, choices = reader.format_batched(
                    [phrase] * len(batch), batch, order=unchanged_order
                )
                values, tokens = reader.option_logits(prompts)
                self.probe_tokens += tokens
                for row, options in zip(values, choices, strict=True):
                    roles = row[:len(ROLES)]
                    centered = roles - roles.mean()
                    position_samples.append(centered)
                    for role, score in zip(options[:-1], centered, strict=True):
                        role_samples[role].append(score)
        self.position_prior = np.mean(position_samples, axis=0)
        self.role_prior = {key: float(np.mean(scores)) for key, scores in role_samples.items()}

    def provenance(self):
        return {
            "neutral_probes": PROBES,
            "offline_input_tokens": self.probe_tokens,
            "offline_rows": len(PROBES) * len(ROLES),
            "position_prior": self.position_prior.tolist(),
            "role_prior": self.role_prior,
        }

    def predict(self, phrases):
        pools = [ROLES] * len(phrases)
        prompts, choices = self.reader.format_batched(phrases, pools)
        scores, tokens = self.reader.option_logits(prompts)
        positions = np.r_[self.position_prior, 0.]
        roles = np.array([[self.role_prior.get(role, 0.) for role in order] for order in choices])
        rev_prompts, rev_choices = self.reader.format_batched(phrases, pools, order=reversed_order)
        reversed_scores, reverse_tokens = self.reader.option_logits(rev_prompts)
        aligned = np.array([
            [row[reverse.index(role)] for role in order]
            for row, order, reverse in zip(reversed_scores, choices, rev_choices, strict=True)
        ])
        comparison = {
            "teacher_all_roles": (scores, 1, tokens),
            "teacher_position_corrected": (scores - positions, 1, tokens),
            "teacher_role_corrected": (scores - roles, 1, tokens),
            "teacher_joint_corrected": (scores - positions - roles, 1, tokens),
            "teacher_reversed_order": (aligned, 1, reverse_tokens),
            "teacher_order_average": ((scores + aligned) / 2, 2, tokens + reverse_tokens),
        }
        outputs = {}
        for arm, (rows, forwards, input_tokens) in comparison.items():
            labels, margin, _ = self.reader.decode_options(rows, choices, input_tokens)
            outputs[arm] = labels, margin, forwards, input_tokens
        return outputs
