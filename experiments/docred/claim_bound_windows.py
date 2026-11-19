"""Compare claim-specific evidence windows at a fixed number of verification pairs."""

import argparse
import os
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import scope_triggered_recheck as cohort
from experiments.docred import work_owner_contrast as source
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("shared_four", "lexical_two", "bound_two")
BUDGETS = (.05, .10)


def two_sentence_windows(case, bind):
    text = case["sentences"]
    if len(text) <= 2:
        return ["\n".join(text)] * 4
    base = source.base
    words, _, idf = base.profile(text)
    role = base.tokens(base.ROLE_CUES[case["role"]])
    role_scores = [base.coverage(role, row, idf) for row in words]
    literal = {}
    for name in dict.fromkeys(name for pair in case["names"] for name in pair):
        terms = tuple(base.tokens(name))
        literal[name] = [max(float(base.contains(row, terms)), .6 * base.coverage(terms, row, idf)) for row in words]
    snippets = []
    for work, person in case["names"]:
        work_scores, person_scores = literal[work], literal[person]

        def score(pair):
            left, right = pair
            coverage = sum(max(signal[left], signal[right])
                           for signal in (work_scores, person_scores, role_scores))
            if bind:
                joint = max(min(work_scores[i], person_scores[i]) * (1 + role_scores[i]) / 2 for i in pair)
                adjacent = min(max(work_scores[i] for i in pair), max(person_scores[i] for i in pair))
                if right != left + 1:
                    adjacent = 0.
                return joint + .5 * adjacent, coverage, -left, -right
            return coverage, -left, -right

        chosen = max(combinations(range(len(text)), 2), key=score)
        snippets.append("\n".join(text[i] for i in chosen))
    return snippets


def observe(case, reader, arm):
    start = perf_counter()
    contexts = ([source.window(case)] * 4 if arm == "shared_four" else
                two_sentence_windows(case, bind=arm == "bound_two"))
    task = source.base.ROLES[case["role"]]
    prompts = [(f"Does the source explicitly say that {person} had the role of {task} on the work {work}?", context)
               for (work, person), context in zip(case["names"], contexts, strict=True)]
    scores, clipped = reader(prompts)
    elapsed = (perf_counter() - start) * 1000
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid verifier output")
    return dict(scores=scores, words=sum(len(text.split()) for text in contexts), ms=elapsed)


def measure(cases, reader, label):
    results = []
    for i, case in enumerate(cases):
        offset = i % len(ARMS)
        results.append((case, {arm: observe(case, reader, arm) for arm in ARMS[offset:] + ARMS[:offset]}))
        if (i + 1) % 60 == 0:
            print(f"{label}: {i+1}/{len(cases)}", flush=True)
    return results


def accepted(record, arm, cutoff):
    case, observations = record
    return sum(s > cutoff and ok for s, ok in zip(observations[arm]["scores"], case["labeled"], strict=True))


def summarize(rows, cutoffs):
    output = {}
    for arm in ARMS:
        cutoff = cutoffs[arm]
        counts = {}
        for group in ("combined", "disjoint", "shared"):
            subset = [r for r in rows if group == "combined" or (sum(r[0]["labeled"]) > 2) == (group == "shared")]
            if subset:
                counts[group] = source.summarize(subset, arm, "absolute", cutoff)
        observations = [arms[arm] for _, arms in rows]
        differences = [(accepted(row, arm, cutoff) - accepted(row, "shared_four", cutoffs["shared_four"])) / 4
                       for row in rows]
        output[arm] = dict(
            groups=counts, mean_words_per_batch=float(np.mean([obs["words"] for obs in observations])),
            p95_ms=float(np.quantile([obs["ms"] for obs in observations], .95)),
            model_pairs=4 * len(rows),
            paired_known_delta=clustered_paired_interval(differences, [case["doc"] for case, _ in rows]))
    return output


def run(original, extended, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    development = list(source.cases(original, extended, "dev"))
    excluded = {doc["title"] for split in ("dev", "test")
                for doc, _ in source.documents(original, extended, split)}
    previous, _ = cohort.holdout(original, extended, excluded)
    new, _ = cohort.holdout(original, extended, excluded | {case["doc"] for case in previous})
    if not new:
        raise ValueError("Need fresh documents for scope comparison")
    tokenizer, model, info = source.reader.load_reader(checkpoint)
    reader = source.reader.reader_scorer(tokenizer, model, system=source.SYSTEM)
    for arm in ARMS:
        observe(development[0], reader, arm)
    rows = {panel: measure(cases, reader, panel)
            for panel, cases in (("development", development), ("previous", previous), ("fresh", new))}
    policies = {}
    for budget in BUDGETS:
        cutoffs = {arm: source.calibrate(rows["development"], arm, "absolute", budget) for arm in ARMS}
        policies[str(budget)] = dict(cutoffs=cutoffs,
                                     panels={name: summarize(items, cutoffs) for name, items in rows.items()})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), cohort_sha256=digest(Path(cohort.__file__)),
                            base_sha256=digest(Path(source.__file__)), checkpoint=info["revision"],
                            distinct_documents={name: len({case["doc"] for case, _ in items})
                                                for name, items in rows.items()},
                            panels={name: len(items) for name, items in rows.items()}, policies=policies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "extended", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
