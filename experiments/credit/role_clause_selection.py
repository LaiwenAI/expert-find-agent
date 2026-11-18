"""Test extracting a credited task before CRediT role normalization."""

import argparse
import json
import os
from pathlib import Path
import re

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.compositional_edits import stable
from experiments.credit.factored_role_edits import choose_batched
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.replay_role_training_transfer import checked_banks, evaluate_roles
from experiments.credit.role_language_router import NISO_SHA256, Router
from experiments.credit.train_role_decision import encode, ridge
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader


LINKS = re.compile(r"\s*;\s*|\s*,?\s*\b(?:while|rather than|instead of|without)\b\s+", re.I)
PUNCTUATION = re.compile(r"\s*;\s*")
ARMS = ("raw_ridge", "first_clause", "last_clause", "cross_encoder_clause", "reader_clause", "direct_glossary")
TASK = ("Identify the quoted fragment that describes the task carried out by the target author. "
        "Ignore tasks performed by a colleague or explicitly excluded. "
        "The correct fragment must contain the credited action.")


def clauses(text):
    separator = PUNCTUATION if ";" in text else LINKS
    return [part.strip(" .,:;-") for part in separator.split(text, maxsplit=1) if part.strip(" .,:;-")]


def reverse_semicolon(text):
    first, separator, second = text.partition(";")
    return f"{second.strip().rstrip('.')}; {first.strip()}." if separator and first.strip() and second.strip() else text


def reader_quotes(reader, texts):
    slots, prompts, options = [], [], []
    resolved = list(texts)
    for slot, text in enumerate(texts):
        parts = clauses(text)
        if len(parts) != 2:
            continue
        if int(stable(text), 16) % 2:
            parts.reverse()
        options.append(parts)
        slots.append(slot)
        prompts.append(
            f"{TASK}\nFull statement: {text}\n"
            f"A: {parts[0]}\nB: {parts[1]}\nC: The full statement is needed.\nAnswer:"
        )
    if prompts:
        labels, _ = reader.classify(prompts)
        for slot, parts, (label, _) in zip(slots, options, labels, strict=True):
            if label != "C":
                resolved[slot] = parts[LABELS.index(label)]
    return resolved


def cross_quotes(reader, texts):
    resolved = []
    for text in texts:
        candidates = clauses(text)
        if len(candidates) == 2:
            if int(stable(text), 16) % 2:
                candidates.reverse()
            scores, truncated = reader.score(text, dict(enumerate(candidates)), [0, 1], task=TASK)
            if truncated:
                raise RuntimeError("Unexpected truncated task excerpt")
            resolved.append(candidates[max(range(2), key=lambda i: scores[i])])
        else:
            resolved.append(text)
    return resolved


def load_groups(folder, audit_path):
    pinned = json.loads(audit_path.read_bytes())
    if pinned["fixtures"]["transfer"] != digest(folder / "manifest.json"):
        raise ValueError("Changed source inventory")
    audit = set(pinned["allowed"])
    manifest = json.loads((folder / "manifest.json").read_bytes())
    pair_path = folder / "pairs.jsonl"
    if digest(pair_path) != manifest["pairs_sha256"]:
        raise ValueError("Changed paired role edits")
    authors = {s["doi"]: s["roles_by_author"] for s in manifest["sources"] if s["doi"] in audit}
    groups = [g for g in map(json.loads, pair_path.read_text().splitlines()) if g["doi"] in authors]
    if set(authors) != audit or {g["doi"] for g in groups} != audit:
        raise ValueError("Inconsistent evaluated sources")
    return groups, authors


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO example bank")
    glossary = json.loads(args.examples.read_bytes())
    banks = checked_banks(args, glossary)
    groups, authors = load_groups(args.transfer, args.audit)
    embedding = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    reranker = Reranker(args.reranker)
    tokenizer, reader_model, reader_provenance = load_reader(args.reader)
    reader = Reader(tokenizer, reader_model)
    docs = [f"{role}: {glossary['definitions'][role]}" for role in ROLES]
    router = Router(docs, None, None, encoder, reader_model, tokenizer, 3)
    original = json.loads(args.training.read_bytes())
    ordered = [[(text, ROLES.index(role)) for text in sorted(original[role],
                key=lambda s: stable("fine-tune-split", role, s))] for role in ROLES]
    fit = [r for examples in ordered for r in examples[:-1]]
    held = [examples[-1] for examples in ordered]
    out = {}
    with torch.inference_mode():
        x = torch.cat([encode(encoder, [t for t, _ in part]) for part in (fit, held)])
        y = torch.tensor([r for _, r in (*fit, *held)], device="cuda")
        bank = ridge(x, y)

        def resolve(texts, method):
            if args.reverse_semicolon:
                texts = [reverse_semicolon(text) for text in texts]
            if method == "direct_glossary":
                sizes = [list(range(len(ROLES))) for _ in texts]
                predictions = [ROLES[i] for i in choose_batched(router, texts, sizes, 0)]
                return predictions
            if method == "first_clause":
                texts = [clauses(t)[0] for t in texts]
            elif method == "last_clause":
                texts = [clauses(t)[-1] for t in texts]
            elif method == "cross_encoder_clause":
                texts = cross_quotes(reranker, texts)
            elif method == "reader_clause":
                texts = reader_quotes(reader, texts)
            return [ROLES[i] for i in (encode(encoder, texts) @ bank.T).argmax(1).tolist()]

        for style, phrases in banks.items():
            out[style] = {}
            example = [phrases[role][0] for role in ROLES[:4]]
            for method in ARMS:
                resolve(example, method)
                out[style][method] = evaluate_roles(lambda texts: resolve(texts, method), phrases, groups, authors)
    write_json(args.output, dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        reverse_semicolon=args.reverse_semicolon,
        script_sha256=digest(Path(__file__)), training_sha256=digest(args.training),
        audit_sha256=digest(args.audit), phrases_sha256=digest(args.phrases),
        embedding=embedding, reader=reader_provenance, reranker=reranker.provenance,
        report=out,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transfer", "audit", "phrases", "examples", "training", "embedding", "reranker", "reader", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--comparison-phrases", type=Path)
    parser.add_argument("--reverse-semicolon", action="store_true")
    run(parser.parse_args())
