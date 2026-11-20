"""Learn who did which work, holding the role and evidence window fixed."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins
from experiments.docred import person_bound_evidence as names
from experiments.docred import work_owner_contrast as source
from experiments.docred.scope_triggered_recheck import TRAIN_SHA256
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("generic", "generic_relative", "geometry", "named", "marked", "marked_contrast")
TASK = names.TASK + " Shared roles are allowed if the source supports each person and work."
SEED = 20261121
CAPS = (.05, .10)


def panel(original, expanded):
    calibration = list(source.cases(original, expanded, "dev"))
    held = list(source.cases(original, expanded, "test"))
    old, new = original / "train_revised.json", expanded / "re2docred_train_re.json"
    if (digest(old), digest(new)) != tuple(TRAIN_SHA256.values()):
        raise ValueError("Unexpected training data hashes")
    updates = {doc["title"]: doc for doc in json.loads(new.read_bytes())}
    excluded = {tuple(names.tokens(row["doc"])) for row in calibration + held}
    cal_titles = {tuple(names.tokens(r["doc"])) for r in calibration}
    held = [r for r in held if tuple(names.tokens(r["doc"])) not in cal_titles]

    def aligned():
        for doc in json.loads(old.read_bytes()):
            if tuple(names.tokens(doc["title"])) in excluded:
                continue
            revised = updates[doc["title"]]
            if doc["sents"] != revised["sents"] or doc["vertexSet"] != revised["vertexSet"]:
                raise ValueError("Training sources differ")
            yield doc, revised

    # All labels in the updated inventory are preserved, including shared
    # credit. Keep only cases with at least one unresolved relation as rival.
    fit = [row for row in source.cases_from_documents(aligned()) if not all(row["labeled"])]
    fit.sort(key=lambda row: row["id"])
    if not fit or not calibration or not held:
        raise ValueError("Empty split")
    return fit, calibration, held


def name_pattern(parts):
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, parts)) + r")(?!\w)", re.IGNORECASE)


def interaction_features(lines, work, person, role):
    needles = [name_pattern([name]) for name in (work, person)]
    cues = name_pattern(names.tokens(names.ROLE_CUES[role]))
    found = np.array([[bool(p.search(line)) for p in [*needles, cues]] for line in lines], dtype=np.float32)
    work_mention, person_mention, role_mention = found.T
    return [
        work_mention.mean(), person_mention.mean(), role_mention.mean(),
        (work_mention * person_mention).mean(), (work_mention * role_mention).mean(),
        (person_mention * role_mention).mean(), (work_mention * person_mention * role_mention).mean(),
    ]


def render(case):
    evidence = source.window(case)
    # Canonical names are supplied candidate identities, not evidence labels.
    all_names = {item.casefold() for pair in case["names"] for item in pair}
    matcher = name_pattern(sorted(all_names, key=lambda item: (-len(item), item)))
    named, marked, geometry = [], [], []
    role = names.ROLES[case["role"]]
    for work, person in case["names"]:
        targets = {work.casefold(): "TARGET WORK", person.casefold(): "TARGET PERSON"}
        anonymized = matcher.sub(lambda m: targets.get(m[0].casefold(), "OTHER ENTITY"), evidence)
        prompt = f"Did {person} have the role of {role} on {work}?"
        named.append((prompt, evidence))
        marked.append(f"Did TARGET PERSON have the role of {role} on TARGET WORK?\n{anonymized}")
        geometry.append(interaction_features(evidence.splitlines(), work, person, case["role"]))
    return named, marked, np.array(geometry, dtype=np.float32)


def observe(case, encoder, reader, arms):
    start = perf_counter()
    raw, marked, geometry = render(case)
    observations = {"geometry": geometry}
    if "named" in arms:
        vectors = encoder.encode([query + "\n" + text for query, text in raw])
        observations["named"] = np.column_stack((vectors, geometry))
    if "marked" in arms:
        vectors = encoder.encode(marked)
        observations["marked"] = np.column_stack((vectors, geometry))
    if "generic" in arms:
        scores, clipped = margins(reader, raw, TASK)
        observations["generic"] = scores
        observations["clipped"] = clipped
    return observations, (perf_counter() - start) * 1000


def fit_head(features, known, contrast):
    inputs = torch.tensor(np.stack(features), dtype=torch.float32, device="cuda")
    labels = torch.tensor(np.stack(known), dtype=torch.bool, device="cuda")
    ground = labels.float()
    torch.manual_seed(SEED)
    head = torch.nn.Linear(inputs.shape[-1], 1, device="cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=.10)
    within_axis = torch.tensor([[abs(a - b) == 2 or (a ^ b) == 1 for b in range(4)]
                                for a in range(4)], dtype=torch.bool, device="cuda")
    binding = labels[:, :, None] & ~labels[:, None, :] & within_axis
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = head(inputs).squeeze(-1)
        objective = F.binary_cross_entropy_with_logits(logits, ground)
        if contrast:
            deltas = logits[:, :, None] - logits[:, None, :]
            objective = objective + .5 * F.softplus(-deltas[binding]).mean()
        objective.backward()
        optimizer.step()
    return head.eval()


def measure(cases, encoder, reader, heads, offset):
    timed = {arm: [] for arm in ARMS}
    measured = {arm: [] for arm in ARMS}
    clipped = 0
    primaries = ("generic", "geometry", "named", "marked")
    for i, case in enumerate([cases[0], *cases]):
        shift = (i + offset) % len(primaries)
        for arm in primaries[shift:] + primaries[:shift]:
            start = perf_counter()
            observed, _ = observe(case, encoder, reader, {arm})
            if arm == "generic":
                values = {arm: observed[arm], "generic_relative": source.rank_scores(observed[arm], "both")}
            else:
                use = ("marked", "marked_contrast") if arm == "marked" else (arm,)
                features = torch.as_tensor(observed[arm], device="cuda")
                with torch.inference_mode():
                    values = {name: heads[name](features).squeeze(-1).cpu().numpy() for name in use}
            json.dumps({name: list(map(float, score)) for name, score in values.items()})
            elapsed = (perf_counter() - start) * 1000
            if i:
                clipped += observed.get("clipped", 0)
                for name, scores in values.items():
                    measured[name].append([float(v) for v in scores])
                    timed[name].append(elapsed)
    return measured, timed, clipped


def summarize(cases, scores, cutoff):
    counts = Counter()
    labels, stamps = [], []
    for row, values in zip(cases, scores, strict=True):
        accepted = [v > cutoff for v in values]
        known = row["labeled"]
        positive = sum(a and b for a, b in zip(accepted, known, strict=True))
        unresolved = sum(a and not b for a, b in zip(accepted, known, strict=True))
        counts["known"] += sum(known)
        counts["unjudged"] += 4 - sum(known)
        counts["retained"] += positive
        counts["unjudged_issued"] += unresolved
        counts["all_known"] += positive == sum(known)
        counts["shared_all_known"] += sum(known) > 2 and positive == sum(known)
        labels.append(positive)
        stamps.append(dict(id=row["id"], acceptance=accepted, labeled=known))
    return dict(counts), labels, stamps


def evaluate(fit, calibration, held, encoder, reader, offset):
    features = {arm: [] for arm in ("geometry", "named", "marked")}
    for row in fit:
        observations, _ = observe(row, encoder, None, {"named", "marked"})
        for arm in features:
            features[arm].append(observations[arm])
    training = [row["labeled"] for row in fit]
    heads = {arm: fit_head(matrix, training, False) for arm, matrix in features.items()}
    heads["marked_contrast"] = fit_head(features["marked"], training, True)
    development, _, dev_clipped = measure(calibration, encoder, reader, heads, offset)
    angles, timers, test_clipped = measure(held, encoder, reader, heads, offset)
    results = {}
    for cap in CAPS:
        route = {}
        reference = None
        for arm in ARMS:
            rivals = sorted((value for case, scores in zip(calibration, development[arm], strict=True)
                             for value, yes in zip(scores, case["labeled"], strict=True) if not yes), reverse=True)
            cutoff = rivals[int(cap * len(rivals))]
            cal, _, _ = summarize(calibration, development[arm], cutoff)
            test, accepted, outcomes = summarize(held, angles[arm], cutoff)
            if reference is None:
                reference = accepted
            route[arm] = dict(cutoff=cutoff, calibration=cal, held=test, outcomes=outcomes,
                              per_case_gain=clustered_paired_interval(
                                  [a - b for a, b in zip(accepted, reference, strict=True)],
                                  [row["doc"] for row in held],
                              ))
        results[str(cap)] = route
    return results, {arm: float(np.quantile(timers[arm], .95)) for arm in ARMS}, [dev_clipped, test_clipped]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "expanded", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    fit, calibration, held = panel(args.original, args.expanded)
    splits = {key: dict(cases=len(rows), titles=len({r["doc"] for r in rows}), known=sum(sum(r["labeled"]) for r in rows))
              for key, rows in (("train", fit), ("calibration", calibration), ("held", held))}
    print(splits, flush=True)
    verified = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    results, latency, clipped = evaluate(fit, calibration, held, encoder, reader, args.offset)
    write_json(args.output, dict(
        splits=splits, input_sha256={"train": TRAIN_SHA256, "original": names.DATA, "expanded": source.UPDATED},
        results=results, p95_ms=latency, reader_clipped=clipped, offset=args.offset,
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        embedding_revision=verified["revision"], reranker_revision=reader.provenance["revision"],
    ))


if __name__ == "__main__":
    main()
