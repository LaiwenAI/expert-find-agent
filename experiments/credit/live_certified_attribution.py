"""Time prefix stopping with a live, source-scoped contributor verifier."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.counterfactual_role_edits import POSITIVE, VERSION, fingerprint
from experiments.credit.evaluate_role_edits import MEANINGS, prompt
from experiments.credit.role_language_router import NISO_SHA256
from experiments.followir.core17_reranking import digest, write_json
from experiments.qasper.paired_reader import load_reader


METHODS = ("full_batch", "fixed_four", "stop_four", "stop_needed")
DEPTHS = (3, 5)
REPEATS = 2


def load(fixture, language):
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    pairs = fixture / "pairs.jsonl"
    if manifest["version"] != VERSION or digest(pairs) != manifest["pairs_sha256"]:
        raise ValueError("Unexpected contributor edit fixture")
    if digest(language) != NISO_SHA256:
        raise ValueError("Unexpected NISO examples")
    phrases = defaultdict(list)
    for row in json.loads(language.read_bytes())["examples"]:
        phrases[row["role"]].append(row["text"].rstrip(". "))
    sources = {p["doi"]: p["roles_by_author"] for p in manifest["sources"] if p["split"] == "test"}
    keys = set()
    for line in pairs.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "test":
            for view in ("before", "after"):
                keys.add((row["doi"], tuple(sorted(row[view]))))
    if len(keys) != 131 or len(sources) != 8:
        raise ValueError("Unexpected held-out inventory")
    cases = []
    for doi, roles in sorted(keys):
        case_id = fingerprint(doi, roles, "live-stopping")
        order = sorted(roles, key=lambda role: fingerprint(case_id, role, "order"))
        options = sorted(MEANINGS, key=lambda meaning: fingerprint(case_id, meaning))
        tasks = [min(phrases[role], key=lambda text: fingerprint(case_id, text)) for role in order]
        question = f"Did the author do both of the following: {tasks[0]}; and {tasks[1]}?"
        authors = list(enumerate(sources[doi], 1))
        gold = [ordinal for ordinal, declarations in authors if set(roles).issubset(declarations)]
        cases.append(dict(id=case_id, doi=doi, authors=authors, options=options, question=question, gold=gold))
    return cases, digest(manifest_path), digest(pairs)


def evaluate(reader, case, method, depth):
    start = perf_counter()
    authors = case["authors"]
    cursor = 0
    positives = []
    decisions = []
    calls = 0
    tokens = 0
    while cursor < len(authors):
        needed = depth - len(positives)
        if method.startswith("stop_") and needed <= 0:
            break
        if method == "full_batch":
            width = len(authors)
        elif method == "stop_needed":
            width = min(4, needed)
        else:
            width = 4
        batch = authors[cursor:cursor + width]
        texts = [prompt(case["question"], ordinal, [(ordinal, roles)], case["options"])
                 for ordinal, roles in batch]
        observed, _ = reader.classify(texts)
        for (ordinal, _), (symbol, length) in zip(batch, observed, strict=True):
            decision = case["options"][LABELS.index(symbol)]
            decisions.append(decision)
            if decision == POSITIVE:
                positives.append(ordinal)
            tokens += length
        cursor += len(batch)
        calls += 1
    citations = [[case["doi"], ordinal] for ordinal in positives[:depth]]
    return dict(citations=citations, decisions=decisions, checks=cursor, tokens=tokens,
                batches=calls, ms=(perf_counter() - start) * 1000)


def summarize(rows):
    report = {}
    for depth in DEPTHS:
        group = [r for r in rows if r["depth"] == depth]
        report[depth] = {}
        repeated = [r for r in group if r["repeat"] == 0]
        by_doi = [r["doi"] for r in repeated]
        for method in METHODS:
            results = [r["arms"][method] for r in group]
            reference = "fixed_four" if method == "stop_four" else "full_batch"
            deltas = [r["arms"][method]["ms"] - r["arms"][reference]["ms"] for r in repeated]
            report[depth][method] = dict(
                measures=len(results), queries=len(repeated),
                same_as_full=sum(r["arms"][method]["citations"] == r["arms"]["full_batch"]["citations"]
                                 for r in group),
                same_as_fixed_four=sum(r["arms"][method]["citations"] == r["arms"]["fixed_four"]["citations"]
                                       for r in group),
                prefix_decisions_changed=sum(
                    r["arms"][method]["decisions"] != r["arms"][reference]["decisions"][:r["arms"][method]["checks"]]
                    for r in group
                ),
                exact_gold=sum([source[1] for source in r["arms"][method]["citations"]] == r["gold"] for r in repeated),
                gold_citations=sum(sum(source[1] in r["gold"] for source in r["arms"][method]["citations"])
                                   for r in repeated),
                returned_citations=sum(len(r["arms"][method]["citations"]) for r in repeated),
                mean_checks=mean(r["checks"] for r in results), mean_batches=mean(r["batches"] for r in results),
                mean_tokens=mean(r["tokens"] for r in results),
                p50_p95_ms=np.quantile([r["ms"] for r in results], [.50, .95]).tolist(),
                latency_delta_ms=clustered_paired_interval(deltas, by_doi, repeats=4000)
            )
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    cases, manifest, pairs = load(args.fixture, args.examples)
    tokenizer, model, provenance = load_reader(args.checkpoint)
    reader = Reader(tokenizer, model)
    for warmup in sorted(cases, key=lambda case: -len(case["authors"]))[:4]:
        for method in METHODS:
            evaluate(reader, warmup, method, DEPTHS[0])
    rows = []
    for i, case in enumerate(cases):
        for repeat in range(REPEATS):
            depths = DEPTHS if (i + repeat) % 2 == 0 else DEPTHS[::-1]
            for depth in depths:
                shift = (i + 2 * repeat + DEPTHS.index(depth)) % len(METHODS)
                arms = {method: evaluate(reader, case, method, depth)
                        for method in METHODS[shift:] + METHODS[:shift]}
                fixed = arms["fixed_four"]
                stopped = arms["stop_four"]
                decisions = stopped["decisions"]
                if fixed["decisions"][:len(decisions)] == decisions and fixed["citations"] != stopped["citations"]:
                    raise AssertionError("Stopping changed the output despite identical prefix decisions")
                rows.append(dict(id=case["id"], doi=case["doi"], gold=case["gold"][:depth],
                                 depth=depth, repeat=repeat, arms=arms))
        if (i + 1) % 25 == 0:
            print("Completed", i + 1, "of", len(cases), flush=True)
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                manifest_sha256=manifest, pairs_sha256=pairs, examples_sha256=NISO_SHA256,
                                script_sha256=digest(Path(__file__)), checkpoint=provenance,
                                report=summarize(rows), rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "examples", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
