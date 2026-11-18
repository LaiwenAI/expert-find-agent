"""Evaluate frozen role models on new wording and linked-source holdouts."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit.compositional_edits import stable
from experiments.credit.counterfactual_role_edits import label
from experiments.credit.cross_source_identity import load
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("ridge70", "ridge84", "head_only", "lora")


def audit(args):
    if args.audit.exists():
        raise FileExistsError(args.audit)
    documents = load(args.xml)
    original = json.loads((args.original / "manifest.json").read_bytes())
    compositions = json.loads((args.fixture / "sources.json").read_bytes())
    transfer = json.loads((args.transfer / "manifest.json").read_bytes())
    forbidden = {row["doi"] for row in original["sources"]} | compositions.keys()
    for row in [*original["sources"], *compositions.values()]:
        if documents[row["doi"]]["source_sha256"] != row["source_sha256"]:
            raise ValueError("Changed source in reference fixture")

    def owners(ids):
        return {author["person"] for doi in ids for author in documents[doi]["authors"] if author["person"]}

    known = owners(forbidden)
    retained = []
    excluded = {}
    for row in transfer["sources"]:
        doi = row["doi"]
        if documents[doi]["source_sha256"] != row["source_sha256"]:
            raise ValueError("Changed transfer XML")
        if doi in forbidden:
            excluded[doi] = "shared_doi"
        elif known & owners([doi]):
            excluded[doi] = "shared_published_identifier"
        else:
            retained.append(doi)
    write_json(args.audit, dict(allowed=sorted(retained), excluded=excluded,
                               fixtures={name: digest(path / "manifest.json")
                                         for name, path in (("original", args.original), ("composition", args.fixture),
                                                            ("transfer", args.transfer))},
                               verified_xml={d: documents[d]["source_sha256"] for d in sorted(forbidden | set(retained))}))


def checked_banks(args, glossary):
    from experiments.credit.role_prototypes import normalized

    published_phrases = getattr(args, "comparison_phrases", None) or args.transfer / "phrases.json"
    new_phrases = args.phrases
    banks = {tag: json.loads(path.read_bytes()) for tag, path in
             (("transfer_wording", published_phrases), ("boundary_wording", new_phrases))}
    seen = {normalized(row["text"]) for row in glossary["examples"]}
    seen.update(normalized(sentence) for group in json.loads(args.training.read_bytes()).values() for sentence in group)
    for name, bank in banks.items():
        if set(bank) != set(ROLES) or any(len(v) != 2 for v in bank.values()):
            raise ValueError(f"Unexpected phrase inventory: {name}")
        current = {normalized(x) for group in bank.values() for x in group}
        if len(current) != 28 or seen & current:
            raise ValueError(f"Unplanned phrase overlap: {name}")
        seen.update(current)
    return banks


def restore_adapter(encoder, weights, head):
    from experiments.credit.train_role_decision import ADAPTER_LAYERS, Adapter

    for index in range(len(encoder.model.layers) - ADAPTER_LAYERS, len(encoder.model.layers)):
        block = encoder.model.layers[index].self_attn
        for name in ("q_proj", "v_proj"):
            wrapper = Adapter(getattr(block, name))
            wrapper.a.copy_(weights[f"{index}.{name}.a"])
            wrapper.b.copy_(weights[f"{index}.{name}.b"])
            setattr(block, name, wrapper)
    head.weight.copy_(weights["head_weight"])
    head.bias.copy_(weights["head_bias"])


def evaluate(encoder, predictor, bank, groups, authors):
    from experiments.credit.train_role_decision import encode

    atomic = [(role, text) for role in ROLES for text in bank[role]]
    predictions = predictor(encode(encoder, [text for _, text in atomic])).argmax(1).tolist()
    results = []
    for group in groups:
        requests = [
            [min(bank[r], key=lambda x: stable(group["id"], r, x)) for r in group[view]]
            for view in ("before", "after")
        ]
        start = perf_counter()
        mapped = [ROLES[i] for i in predictor(encode(encoder, sum(requests, []))).argmax(1).tolist()]
        after = perf_counter()
        roles = (mapped[:2], mapped[2:])
        paper = authors[group["doi"]]
        changed = [label(paper[group["changed_author"] - 1], match) for match in roles]
        stable_pair = [label(paper[group["stable_author"] - 1], match) for match in roles]
        results.append(dict(id=group["id"], doi=group["doi"], changed=changed, stable=stable_pair,
                            ms=(after-start)*1000))
    records = list(zip(groups, results, strict=True))
    return dict(atomic_correct=sum(ROLES[i] == role for i, (role, _) in zip(predictions, atomic, strict=True)),
                atomic_total=len(atomic),
                joint_groups=sum(r["changed"] == g["changed_labels"] and r["stable"] == g["stable_labels"]
                                 for g, r in records),
                change_pairs=sum(r["changed"] == g["changed_labels"] for g, r in records),
                stable_pairs=sum(r["stable"] == g["stable_labels"] for g, r in records),
                false_promotions=sum(a == "declared" and b != "declared"
                                     for g, r in records for key in ("changed", "stable")
                                     for a, b in zip(r[key], g[f"{key}_labels"], strict=True)),
                groups=len(groups), p95_ms=float(np.quantile([r["ms"] for r in results], .95)), rows=results)


def run(args):
    import torch
    from safetensors.torch import load_file

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.train_role_decision import SEED, encode, head_from, inputs, ridge
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    rhs = json.loads(args.audit.read_bytes())
    if not rhs["allowed"] or rhs["fixtures"]["transfer"] != digest(args.transfer / "manifest.json"):
        raise ValueError("Changed held-out publication inventory")
    if rhs["fixtures"]["composition"] != digest(args.fixture / "manifest.json"):
        raise ValueError("Changed reference audit scope")
    transfer = json.loads((args.transfer / "manifest.json").read_bytes())
    path = args.transfer / "pairs.jsonl"
    if digest(path) != transfer["pairs_sha256"]:
        raise ValueError("Changed matched groups")
    authors = {row["doi"]: row["roles_by_author"] for row in transfer["sources"] if row["doi"] in rhs["allowed"]}
    groups = [row for row in map(json.loads, path.read_text().splitlines()) if row["doi"] in authors]
    if set(authors) != set(rhs["allowed"]) or {g["doi"] for g in groups} != set(authors):
        raise ValueError("Inconsistent held-out source inventory")
    args.seed = SEED
    fit, valid, glossary, _, _ = inputs(args)
    banks = checked_banks(args, glossary)
    run = json.loads((args.checkpoints / "results.json").read_bytes())
    provenance = verify_checkpoint(args.embedding)
    if (provenance != run["embedding"] or digest(args.training) != run["training_sha256"]
            or digest(args.fixture / "manifest.json") != run["fixture_sha256"]):
        raise ValueError("Model source or training provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    report = defaultdict(dict)
    with torch.inference_mode():
        x = encode(encoder, [text for text, _ in fit])
        y = torch.tensor([role for _, role in fit], device="cuda")
        xv = encode(encoder, [text for text, _ in valid])
        yv = torch.tensor([role for _, role in valid], device="cuda")
        base = ridge(x, y)
        full = ridge(torch.cat((x, xv)), torch.cat((y, yv)))
        head = head_from(base)
        for method in METHODS:
            if method == "ridge70":
                predictor = lambda x: x @ base.T
            elif method == "ridge84":
                predictor = lambda x: x @ full.T
            else:
                path = args.checkpoints / f"{method}.safetensors"
                if digest(path) != run["metadata"][method]["checkpoint_sha256"]:
                    raise ValueError(f"Changed checkpoint: {method}")
                weights = load_file(str(path), device="cuda")
                if method == "lora":
                    restore_adapter(encoder, weights, head)
                else:
                    head.weight.copy_(weights["head_weight"])
                    head.bias.copy_(weights["head_bias"])
                predictor = head
            for name, bank in banks.items():
                report[name][method] = evaluate(encoder, predictor, bank, groups, authors)
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 script_sha256=digest(Path(__file__)), audit_sha256=digest(args.audit),
                                 native_report_sha256=digest(args.checkpoints/"results.json"),
                                 phrases_sha256=digest(args.phrases), groups=len(groups),
                                 source_count=len(authors), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit")
    for name in ("original", "transfer", "fixture", "audit"):
        check.add_argument(name, type=Path)
    check.add_argument("xml", type=Path, nargs="+")
    replay = sub.add_parser("run")
    for name in ("transfer", "fixture", "audit", "phrases", "examples", "training", "embedding", "checkpoints", "output"):
        replay.add_argument(name, type=Path)
    args = parser.parse_args()
    (audit if args.command == "audit" else run)(args)
