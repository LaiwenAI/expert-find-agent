"""Route glossary reading by whether a role can change a cross-paper decision."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.compositional_edits import VERSION, stable
from experiments.credit.factored_role_edits import candidate_lists, choose_batched
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, Router
from experiments.credit.role_prototypes import encode_tasks, train_banks
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("ridge", "always_read", "shortlist_read", "evidence_route")
WIDTHS = (1, 2, 3, 4, 6, 8, len(ROLES))


def evidence(sources, row):
    authors = [sources[doi]["authors"][ordinal - 1]
               for doi, ordinal in zip(row["sources"], row["ordinals"], strict=True)]
    hidden = {"left_withheld": 0, "right_withheld": 1}.get(row["view"])
    identity = None if hidden is not None else authors[0]["person"] == authors[1]["person"]
    by_slot = [set(author["roles"]) if i != hidden else None for i, author in enumerate(authors)]
    return identity, [by_slot[slot] for slot, _ in row["requirements"]]


def conclusion(identity, observations):
    if identity is False or False in observations:
        return "rejected"
    if identity is None or None in observations:
        return "unresolved"
    return "supported"


def render(row, phrases):
    return [min(phrases[role], key=lambda p: stable(row["group"], role, p))
            for _, role in row["requirements"]]


def calibrate(router, bank, sources, rows, phrases):
    atoms = {}
    for row in rows:
        identity, declarations = evidence(sources, row)
        if identity is False:
            continue
        for atom, (_, role), record in zip(render(row, phrases), row["requirements"], declarations, strict=True):
            if record is not None:
                expected = ROLES.index(role)
                if atom in atoms and atoms[atom] != expected:
                    raise ValueError("Conflicting taxonomy phrase")
                atoms[atom] = expected
    if not atoms:
        raise ValueError("No eligible calibration phrases")
    ranks = np.argsort(-(encode_tasks(router.embedder, list(atoms)) @ bank.T), axis=1, kind="stable").tolist()
    coverage = {}
    for width in WIDTHS:
        sets = candidate_lists(router, atoms, ranks, width)
        coverage[width] = sum(role in choices for role, choices in zip(atoms.values(), sets, strict=True))
    width = next(w for w, count in coverage.items() if count == len(atoms))
    return width, dict(unique_phrases=len(atoms), coverage=coverage)


def read_glossaries(router, phrases, indices, rotation, options):
    # Keep the same glossary order per original atom, even in a smaller batch.
    orders = []
    for local, original in enumerate(indices):
        roles = options[original] if options is not None else list(range(len(ROLES)))
        shift = (rotation + original - local) % len(roles)
        orders.append(roles[shift:] + roles[:shift])
    return choose_batched(router, [phrases[i] for i in indices], orders, 0)


@torch.inference_mode()
def query(router, bank, sources, row, phrases, method, width, rotation):
    start = perf_counter()
    identity, declarations = evidence(sources, row)
    visible = [i for i, record in enumerate(declarations) if record is not None]
    vals = [None] * len(phrases)
    shortlists = None
    mapped = {}
    read = []
    if identity is not False and visible:
        if method != "always_read":
            scores = encode_tasks(router.embedder, [phrases[i] for i in visible]) @ bank.T
            ranks = np.argsort(-scores, axis=1, kind="stable").tolist()
        if method == "ridge":
            mapped.update((i, rank[0]) for i, rank in zip(visible, ranks, strict=True))
        elif method == "always_read":
            read = visible
        else:
            shortlists = dict(zip(visible, candidate_lists(router, [phrases[i] for i in visible],
                                                         ranks, width), strict=True))
            if method == "shortlist_read":
                read = visible
            else:
                for i, options in shortlists.items():
                    answers = {ROLES[role] in declarations[i] for role in options}
                    if len(answers) == 1:
                        vals[i] = answers.pop()
                    else:
                        read.append(i)
                if False in vals:
                    read = []
        if read:
            options = shortlists if method == "shortlist_read" else None
            choices = read_glossaries(router, phrases, read, rotation, options)
            mapped.update(zip(read, choices, strict=True))
        for i, role in mapped.items():
            vals[i] = ROLES[role] in declarations[i]
    return dict(label=conclusion(identity, vals), read_atoms=read, options=shortlists,
                mapped=mapped, ms=(perf_counter() - start) * 1000)


def summarize(rows):
    out = {}
    for view in ("all", *sorted({r["view"] for r in rows})):
        subset = [r for r in rows if view == "all" or r["view"] == view]
        out[view] = {}
        for method in METHODS:
            results = [r["arms"][method] for r in subset]
            out[view][method] = dict(
                requests=len(results),
                correct=sum(a["label"] == r["reference"] for a, r in zip(results, subset, strict=True)),
                unsupported=sum(a["label"] == "supported" and r["reference"] != "supported"
                                for a, r in zip(results, subset, strict=True)),
                lost_supported=sum(a["label"] != "supported" and r["reference"] == "supported"
                                   for a, r in zip(results, subset, strict=True)),
                unresolved_as_rejected=sum(a["label"] == "rejected" and r["reference"] == "unresolved"
                                           for a, r in zip(results, subset, strict=True)),
                read_requests=sum(bool(a["read_atoms"]) for a in results),
                read_atoms=sum(len(a["read_atoms"]) for a in results),
                mean_ms=mean(a["ms"] for a in results),
                p95_ms=float(np.quantile([a["ms"] for a in results], .95)),
            )
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
    rows = json.loads((args.fixture / "queries.json").read_bytes())
    glossary = json.loads(args.examples.read_bytes())
    phrases = defaultdict(list)
    for row in glossary["examples"]:
        phrases[row["role"]].append(row["text"])
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    docs = [f"{role}: {glossary['definitions'][role]}" for role in ROLES]
    tokenizer, reader, reader_provenance = load_reader(args.reader)
    tokenizer.padding_side = "left"
    with TemporaryDirectory() as temp:
        lexical = build_lexical({text: i for i, text in enumerate(docs)}, Path(temp))
    router = Router(docs, lexical, encoder.encode(docs), encoder, reader, tokenizer, 3)
    banks, training = train_banks(args.training, router, glossary["examples"])
    bank = banks["ridge"]
    width, calibration = calibrate(router, bank, sources, [r for r in rows if r["split"] == "calibration"], phrases)
    results = []
    for i, row in enumerate(r for r in rows if r["split"] == "test"):
        atoms = render(row, phrases)
        rotation = int(stable(row["group"], "glossary"), 16) % len(ROLES)
        if not i:
            for method in METHODS:
                query(router, bank, sources, row, atoms, method, width, rotation)
        offset = i % len(METHODS)
        arms = {m: query(router, bank, sources, row, atoms, m, width, rotation)
                for m in METHODS[offset:] + METHODS[:offset]}
        results.append(dict(row, arms=arms))
    write_json(args.output, dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        fixture_sha256=digest(manifest_path), training_sha256=digest(args.training),
        script_sha256=digest(Path(__file__)), embedding=provenance, reader=reader_provenance,
        training=training, calibration=calibration, width=width, report=summarize(results), rows=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "examples", "training", "embedding", "reader", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
