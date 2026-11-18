"""Replay published requirement edits with frozen open readers."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.compositional_edits import VERSION, decide, stable
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256
from experiments.credit.role_prototypes import encode_tasks, train_banks
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader


MEANINGS = {
    "supported": "The two records are confirmed to name the same person and all three tasks are declared.",
    "rejected": "A complete record omits a required task, or the records identify different people.",
    "unresolved": "No visible contradiction, but a required record or identity confirmation is missing.",
}
METHODS = ("joint_reader", "factored_ridge")


def prompt(sources, row, phrases, options):
    records = []
    named = []
    for slot, (doi, ordinal) in enumerate(zip(row["sources"], row["ordinals"], strict=True)):
        article = "AB"[slot]
        author = sources[doi]["authors"][ordinal - 1]
        unavailable = row["view"] == ("left_withheld", "right_withheld")[slot]
        named.append(None if unavailable else author["person"])
        content = "unavailable" if unavailable else ", ".join(author["roles"])
        records.append(f"Publication {article}, target author: {content}.")
    identity = "unknown" if None in named else "same person" if named[0] == named[1] else "different people"
    requirements = [f"Publication {'AB'[slot]}: {phrase}"
                    for (slot, _), phrase in zip(row["requirements"], phrases, strict=True)]
    return "\n".join([
        "Evaluate all three requirements for the two target author records.",
        "A visible contradiction is enough for rejection even if another record is missing.",
        *(f"{token}: {MEANINGS[value]}" for token, value in zip(LABELS, options, strict=True)),
        f"Publisher identity check: {identity}.", *records, *requirements, "Answer:",
    ])


def evaluate(method, sources, row, phrases, options, reader, encoder, bank):
    start = perf_counter()
    if method == "joint_reader":
        selected, _ = reader.classify([prompt(sources, row, phrases, options)])
        label = options[LABELS.index(selected[0][0])]
        mappings = None
    else:
        scores = encode_tasks(encoder, phrases) @ bank.T
        mappings = [ROLES[i] for i in scores.argmax(axis=1)]
        inferred = [[slot, role] for (slot, _), role in zip(row["requirements"], mappings, strict=True)]
        label, _, _ = decide(sources, dict(row, requirements=inferred), "author_scoped")
    return dict(label=label, mapped_roles=mappings, ms=(perf_counter() - start) * 1000)


def summarize(rows):
    cohorts = {"all": rows}
    for view in sorted({r["view"] for r in rows}):
        cohorts[view] = [r for r in rows if r["view"] == view]
    out = {}
    for view, subset in cohorts.items():
        report = {}
        for method in METHODS:
            measures = [r["arms"][method] for r in subset]
            report[method] = dict(
                requests=len(measures),
                correct=sum(a["label"] == r["reference"] for a, r in zip(measures, subset, strict=True)),
                unsupported=sum(a["label"] == "supported" and r["reference"] != "supported"
                                for a, r in zip(measures, subset, strict=True)),
                lost_supported=sum(a["label"] != "supported" and r["reference"] == "supported"
                                   for a, r in zip(measures, subset, strict=True)),
                unresolved_as_rejected=sum(a["label"] == "rejected" and r["reference"] == "unresolved"
                                           for a, r in zip(measures, subset, strict=True)),
                mean_ms=mean(a["ms"] for a in measures),
                p95_ms=float(np.quantile([a["ms"] for a in measures], .95)),
            )
        out[view] = report
    return out


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    manifest_path = args.fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["version"] != VERSION or any(digest(args.fixture / n) != s for n, s in manifest["files"].items()):
        raise ValueError("Changed benchmark fixture")
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO examples")
    sources = json.loads((args.fixture / "sources.json").read_bytes())
    rows = [r for r in json.loads((args.fixture / "queries.json").read_bytes()) if r["split"] == "test"]
    if len(rows) != manifest["queries"]["test"]:
        raise ValueError("Unexpected test inventory")
    glossary = json.loads(args.examples.read_bytes())
    by_role = defaultdict(list)
    for row in glossary["examples"]:
        by_role[row["role"]].append(row["text"])
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    definitions = encoder.encode([glossary["definitions"][role] for role in ROLES])
    context = SimpleNamespace(embedder=encoder, vectors=definitions)
    banks, training = train_banks(args.training, context, glossary["examples"])
    tokenizer, model, reader_provenance = load_reader(args.reader)
    reader = Reader(tokenizer, model)
    result = []
    for i, row in enumerate(rows):
        phrases = [min(by_role[role], key=lambda x: stable(row["group"], role, x))
                   for _, role in row["requirements"]]
        choices = sorted(MEANINGS, key=lambda value: stable(row["group"], value))
        if not i:
            for method in METHODS:
                evaluate(method, sources, row, phrases, choices, reader, encoder, banks["ridge"])
        order = METHODS if i % 2 == 0 else METHODS[::-1]
        arms = {m: evaluate(m, sources, row, phrases, choices, reader, encoder, banks["ridge"]) for m in order}
        result.append(dict(row, arms=arms))
    write_json(args.output, dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        fixture_sha256=digest(manifest_path), training_sha256=digest(args.training),
        script_sha256=digest(Path(__file__)), embedding=provenance, reader=reader_provenance,
        training=training, report=summarize(result), rows=result,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "examples", "training", "embedding", "reader", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
