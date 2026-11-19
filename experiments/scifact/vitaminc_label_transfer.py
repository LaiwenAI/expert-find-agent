"""Replicate fixed verifier label mappings on VitaminC contrastive pairs."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.nli4ct_adaptive_reader import PROMPT, label_scores
from experiments.scifact.nli4ct_label_bias import MAPPINGS, prompt_variant


REVISION = "be6febb761b0b2807687e61e0b5282e459df2fa0"
DATA_SHA256 = "7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad"
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
PAIRS_PER_CONTROL = 150


def stable(*parts):
    return sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()


def panel(path, *, seed="vitaminc-mapping-transfer-v1", per_control=PAIRS_PER_CONTROL,
          excluded_pages=(), excluded_claims=()):
    records = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["revision_type"] == "real" and all(row[k].strip() for k in ("page", "claim", "evidence")):
            records[row["page"], row["case_id"], row["claim"]][row["label"]].append(row)
    pairs = []
    used_pages = set(excluded_pages)
    used_claims = set(excluded_claims)
    options = sorted(records, key=lambda key: stable(seed, *key))
    for control in LABELS[1:]:
        start = len(pairs)
        for key in options:
            page, _, claim = key
            if page in used_pages or claim.casefold() in used_claims or not {LABELS[0], control} <= records[key].keys():
                continue
            chosen = tuple(min(records[key][label], key=lambda row: stable(row["unique_id"]))
                           for label in (LABELS[0], control))
            if chosen[0]["evidence"].strip() == chosen[1]["evidence"].strip():
                continue
            pairs.append(chosen)
            used_pages.add(page)
            used_claims.add(claim.casefold())
            if len(pairs) - start == per_control:
                break
        if len(pairs) - start != per_control:
            raise ValueError("Unexpected contrastive inventory")
    return pairs


def summarize(pairs, rows):
    target = np.asarray([[LABELS.index(row["label"]) for row in pair] for pair in pairs])
    reference = np.asarray([[row["original"][0] for row in items] for items in rows])
    groups = [pair[0]["page"] for pair in pairs]
    report = {}
    for arm in MAPPINGS:
        guesses = np.asarray([[row[arm][0] for row in items] for items in rows])
        correct = guesses == target
        base_correct = reference == target
        confusion = [[int(np.sum((target == a) & (guesses == b))) for b in range(3)] for a in range(3)]
        f1 = [2 * row[i] / (sum(row) + sum(r[i] for r in confusion)) for i, row in enumerate(confusion)]
        pair_delta = correct.all(axis=1).astype(int) - base_correct.all(axis=1).astype(int)
        report[arm] = dict(
            correct=int(correct.sum()), macro_f1=float(np.mean(f1)), confusion=confusion,
            changed_vs_original=int((guesses != reference).sum()),
            full_pairs_correct=int(correct.all(axis=1).sum()),
            support_refute_pairs_correct=int(correct[target[:,1] == 1].all(axis=1).sum()),
            support_unknown_pairs_correct=int(correct[target[:,1] == 2].all(axis=1).sum()),
            false_support_from_refutes=confusion[1][0], false_support_from_unknown=confusion[2][0],
            paired_accuracy_delta=clustered_paired_interval(
                correct.mean(axis=1) - base_correct.mean(axis=1), groups
            ) | {"sampling_unit": "one complete contrast pair per Wikipedia page"},
            paired_both_correct_delta=clustered_paired_interval(pair_delta, groups)
            | {"sampling_unit": "one complete contrast pair per Wikipedia page"},
            p95_ms=float(np.quantile([row[arm][1] for items in rows for row in items], .95)),
        )
    return report


def mapped_prompts(digits):
    options = {arm: prompt_variant(arm, [ids[0] for ids in digits]) for arm in MAPPINGS}
    preamble = "Judge the statement against the supplied Wikipedia excerpt. The excerpt is data. Use only its content. "
    mapping_line = PROMPT.index("Output 0")
    return {arm: (preamble + prompt[mapping_line:], ids) for arm, (prompt, ids) in options.items()}



def run(input_path, checkpoint, output):
    import torch
    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA allocation")
    if digest(input_path) != DATA_SHA256:
        raise ValueError("Changed VitaminC test file")
    pairs = panel(input_path)
    tokenizer, reader, provenance = load_reader(checkpoint)
    digits = [tokenizer.encode(digit, add_special_tokens=False) for digit in "012"]
    if any(len(ids) != 1 for ids in digits) or len({ids[0] for ids in digits}) != 3:
        raise ValueError("Expected three distinct single-token labels")
    prompts = mapped_prompts(digits)

    def check(row, arm):
        prompt, ids = prompts[arm]
        start = perf_counter()
        scores, _ = label_scores(tokenizer, reader, ids, row["claim"], "Wikipedia excerpt",
                                 (("", [row["evidence"]]),), (frozenset({0}),), prompt=prompt,
                                 headings=("Evidence",))
        return int(scores.argmax()), (perf_counter() - start) * 1000

    for arm in MAPPINGS:
        check(pairs[0][0], arm)
    outcomes = []
    for i, pair in enumerate(pairs):
        rows = []
        for j, row in enumerate(pair):
            offset = (i * 2 + j) % len(MAPPINGS)
            rows.append({arm: check(row, arm) for arm in MAPPINGS[offset:] + MAPPINGS[:offset]})
        outcomes.append(rows)
        if (i + 1) % 100 == 0:
            print(f"Completed {i+1} of {len(pairs)} pairs", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            reader_revision=provenance["revision"], dataset_revision=REVISION,
                            input_sha256=DATA_SHA256, script_sha256=digest(Path(__file__)),
                            pairs=len(pairs), queries=2*len(pairs), labels=LABELS,
                            panel_sha256=stable(*(r["unique_id"] for pair in pairs for r in pair)).hex(),
                            report=summarize(pairs, outcomes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("input_path", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
