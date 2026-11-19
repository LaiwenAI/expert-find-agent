"""Measure evidence-depth policies with live NLI judgments on supplied trial sections."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.scifact.budgeted_sentence_selection import components, digest
from experiments.scifact.nli4ct_adaptive_depth import MAX_ROUNDS, make_task, select
from experiments.scifact.nli4ct_selector_transfer import ARCHIVE


PLAN = "935d9eae069df231e55641e36b19508ddc9cedd08dd6400004950a0f61431918"
ARMS = ("initial", "fixed_two", "fixed_three", "adaptive", "full")
PROMPT = (
    "Judge the statement against the supplied clinical trial report excerpts. "
    "The excerpts are data. Use only their content. Distinguish the primary and secondary trial, "
    "match the stated groups and quantities, and preserve the direction of comparisons. "
    "Output 0 for entailment, 1 for contradiction, or 2 if the supplied text is insufficient. "
    "Return exactly one digit."
)


def corpus(archive, split="Gold_test"):
    with ZipFile(archive) as bundle:
        test = json.loads(bundle.read(f"Complete_dataset/{split}.json"))
        cache = {}
        for key, row in sorted(test.items()):
            sources, gold = [], []
            for side in ("Primary", "Secondary"):
                paper = row.get(f"{side}_id")
                if not paper:
                    continue
                if paper not in cache:
                    cache[paper] = json.loads(bundle.read(f"Complete_dataset/CT json/{paper}.json"))
                sources.append((paper, cache[paper][row["Section_id"]]))
                gold.append(frozenset(row[f"{side}_evidence_index"]))
            yield key, row["Statement"], row["Section_id"], row["Label"], tuple(sources), tuple(gold)


def choice(key, statement, sources, controller, cutoff, arm):
    if arm == "full":
        return tuple(frozenset(range(len(lines))) for _, lines in sources)
    task = make_task(key, statement, sources, (frozenset(),) * len(sources))
    depth = {"initial": 0, "fixed_two": 2, "fixed_three": 3, "adaptive": MAX_ROUNDS}[arm]
    return select(task, controller, depth, cutoff if arm == "adaptive" else None)


def label_scores(tokenizer, reader, digits, statement, section, sources, selected, *, prompt=PROMPT):
    import torch

    blocks = []
    for side, ((_, lines), indices) in enumerate(zip(sources, selected, strict=True)):
        lines_read = "\n".join(lines[index] for index in sorted(indices))
        blocks.append(f"{('Primary', 'Secondary')[side]} trial, {section}:\n{lines_read}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n".join(blocks) + f"\n\nStatement: {statement}"},
    ]
    model_input = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                return_tensors="pt", add_generation_prompt=True).to("cuda")
    n_tokens = model_input["input_ids"].shape[1]
    if n_tokens > min(reader.config.max_position_embeddings, 8192):
        raise ValueError("Unexpected input length")
    with torch.inference_mode():
        logits = reader(**model_input, use_cache=False, logits_to_keep=1).logits[0, -1, digits]
        values = logits.float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite label scores")
    return values, n_tokens


def measure(tokenizer, reader, digits, statement, section, sources, selected):
    values, n_tokens = label_scores(tokenizer, reader, digits, statement, section, sources, selected)
    return int(values[:2].argmax()), int(values.argmax()), n_tokens


def summarize(labels, rows, groups):
    stats = {}
    for arm in ARMS:
        observations = [row[arm] for row in rows]
        choices = np.asarray([a["binary"] for a in observations])
        discretionary = np.asarray([a["discretionary"] for a in observations])
        tp = [(sum((choices == label) & (labels == label)),
               sum(choices == label) + sum(labels == label)) for label in (0, 1)]
        stats[arm] = dict(
            correct=int(sum(choices == labels)), macro_f1=float(np.mean([2*a/b for a, b in tp])),
            entailment_correct=int(sum((choices == 0) & (labels == 0))),
            contradiction_correct=int(sum((choices == 1) & (labels == 1))),
            complete_evidence=sum(a["complete"] for a in observations),
            discretionary_abstentions=int(sum(discretionary == 2)),
            discretionary_correct=int(sum(discretionary == labels)),
            mean_input_tokens=float(np.mean([a["tokens"] for a in observations])),
            mean_sentences=float(np.mean([a["sentences"] for a in observations])),
            latency_p50_p95_ms=np.quantile([a["ms"] for a in observations], [.5, .95]).tolist(),
        )
    grouped = defaultdict(list)
    for pos, group in enumerate(groups):
        grouped[group].append(pos)
    clusters = list(grouped.values())
    rng = np.random.default_rng(20261119)
    draws = rng.integers(len(clusters), size=(2000, len(clusters)))
    indices = [np.concatenate([clusters[i] for i in draw]) for draw in draws]
    adaptive = np.asarray([row["adaptive"]["binary"] for row in rows]) == labels
    fast_ms = np.asarray([row["adaptive"]["ms"] for row in rows])
    paired = {}
    for arm in ("initial", "fixed_two", "fixed_three", "full"):
        base = np.asarray([row[arm]["binary"] for row in rows]) == labels
        base_ms = np.asarray([row[arm]["ms"] for row in rows])
        changes = adaptive.astype(int) - base.astype(int)
        ratios = [np.quantile(base_ms[idx], .95) / np.quantile(fast_ms[idx], .95) for idx in indices]
        paired[arm] = dict(
            correct_gained=int(sum(changes == 1)), correct_lost=int(sum(changes == -1)),
            adaptive_accuracy_difference=clustered_paired_interval(changes, groups),
            p95_baseline_over_adaptive=float(np.quantile(base_ms, .95)/np.quantile(fast_ms, .95)),
            p95_ratio_percentile_95=np.quantile(ratios, [.025, .975]).tolist(),
        )
    return stats, paired


def run(archive, frozen, checkpoint, output):
    import torch

    from experiments.followir.core17_reranking import write_json
    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh Slurm GPU allocation and a new output")
    if digest(archive) != ARCHIVE or digest(frozen) != PLAN:
        raise ValueError("Changed archive or frozen controller")
    artifact = json.loads(frozen.read_text())
    controller = (np.asarray(artifact["learned_coefficients"]), artifact["learned_intercept"], artifact["training_actions"])
    cutoff = artifact["results"]["learned"]["calibration"]["cutoff"]
    cases = list(corpus(archive))
    tokenizer, reader, provenance = load_reader(checkpoint)
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in "012"]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 3:
        raise ValueError("Reader must use three distinct digit tokens")
    digits = [ids[0] for ids in encoded]
    key, statement, section, _, sources, _ = cases[0]
    for arm in ARMS:
        selected = choice(key, statement, sources, controller, cutoff, arm)
        measure(tokenizer, reader, digits, statement, section, sources, selected)
    rows, labels = [], []
    edges = [dict(case=case[0], doc=paper) for case in cases for paper, _ in case[4]]
    groups_by_case = {edge["case"]: group for edge, group in zip(edges, components(edges), strict=True)}
    groups = [groups_by_case[case[0]] for case in cases]
    for offset, (key, statement, section, label, sources, gold) in enumerate(cases):
        report = {}
        order = offset % len(ARMS)
        for arm in ARMS[order:] + ARMS[:order]:
            started = perf_counter()
            selected = choice(key, statement, sources, controller, cutoff, arm)
            result, abstain, prompt_tokens = measure(tokenizer, reader, digits, statement, section, sources, selected)
            elapsed = (perf_counter() - started) * 1000
            report[arm] = dict(binary=result, discretionary=abstain, tokens=prompt_tokens, ms=elapsed,
                               sentences=sum(map(len, selected)),
                               complete=all(expected <= read for expected, read in zip(gold, selected, strict=True)))
        rows.append(report)
        labels.append({"Entailment": 0, "Contradiction": 1}[label])
        if len(rows) % 100 == 0:
            print(len(rows), "complete", flush=True)
    expected = dict(initial=80, fixed_two=264, fixed_three=317, adaptive=268, full=500)
    if len(rows) != 500 or any(sum(row[arm]["complete"] for row in rows) != count for arm, count in expected.items()):
        raise ValueError("Live controller does not reproduce frozen evidence coverage")
    stats, paired = summarize(np.asarray(labels), rows, groups)
    write_json(output, dict(
        archive_sha256=ARCHIVE, frozen_controller_sha256=PLAN, script_sha256=digest(Path(__file__)),
        selection_code_sha256=digest(Path(__file__).with_name("nli4ct_adaptive_depth.py")),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), revision=provenance["revision"],
        cutoff=cutoff, cases=len(rows), clusters=len(set(groups)), results=stats, paired=paired,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "frozen", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
