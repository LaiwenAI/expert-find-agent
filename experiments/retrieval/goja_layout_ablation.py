"""Remove line breaks without changing candidate spans in the GOJA test."""

import argparse
from hashlib import sha256
import os
from pathlib import Path

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.goja_experience_binding import encode_cases, fit, make_cases


def run(archive, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh result path on a Slurm allocation")
    cases = list(make_cases(archive))
    original = encode_cases(cases, None)
    flattened = encode_cases([
        (group, text.replace("\n", " ").replace("\r", " "), anchor, menu, correct)
        for group, text, anchor, menu, correct in cases
    ], None)
    train = [i for i, (group, _, _) in enumerate(original) if int(group[:8], 16) % 10 < 6]
    test = [i for i, (group, _, _) in enumerate(original) if int(group[:8], 16) % 10 >= 6]
    natural = fit([original[i] for i in train], [0, 1, 2, 3])
    flat = fit([flattened[i] for i in train], [0, 1, 2, 3])

    variants = {
        "nearest": (original, lambda f: f[:, 0]),
        "line_first": (original, lambda f: f[:, 0] + 10 * f[:, 1]),
        "fit_original_apply_original": (original, natural),
        "fit_original_apply_flat": (flattened, natural),
        "fit_flat_apply_flat": (flattened, flat),
    }
    answers = {name: [int(fn(rows[i][1]).argmax() == rows[i][2]) for i in test]
               for name, (rows, fn) in variants.items()}
    named = "fit_original_apply_original"
    groups = [original[i][0] for i in test]
    results = {
        name: {
            "correct": sum(hits),
            "paired_vs_original_layout": clustered_paired_interval(
                [a - b for a, b in zip(hits, answers[named], strict=True)], groups
            ),
        }
        for name, hits in answers.items()
    }
    assert all(r[1][:, 1].all() for r in flattened)
    write_json(output, dict(archive=digest(archive), script=digest(Path(__file__)),
                            test_anchors=len(test), test_docs=len(set(groups)), results=results,
                            split_signature=sha256("".join(groups).encode()).hexdigest(),
                            job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
