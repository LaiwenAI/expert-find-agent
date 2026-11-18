"""Generate ownership swaps and test them as supervision for a frozen role encoder."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.credit.compositional_edits import stable
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json


NEAR = ((0, 5), (2, 8), (1, 4), (3, 7), (6, 9), (10, 11), (12, 13))
OTHER = tuple((i, i + 7) for i in range(7))
MARKERS = (
    ("The selected author performed", "A colleague performed"),
    ("Assigned to this contributor", "Assigned to a different contributor"),
)


def descriptions(path):
    rows = json.loads(path.read_bytes())
    if set(rows) != set(ROLES) or any(len(rows[r]) != 6 for r in ROLES):
        raise ValueError("Changed training description inventory")
    return {r: [s.rstrip(". ") for s in rows[r]] for r in ROLES}


def generate(path):
    seeds = descriptions(path)
    controls = [
        dict(role=role, text=f'{own}: "{text}".')
        for role in ROLES for text in seeds[role] for own, _ in MARKERS
    ]
    result = {"single_clause": controls}
    for mode, matches in (("near_role_swap", NEAR), ("other_role_swap", OTHER)):
        rows = []
        for a, b in matches:
            roles = ROLES[a], ROLES[b]
            for index, pair in enumerate(zip(*(seeds[r] for r in roles), strict=True)):
                for template, (own, peer) in enumerate(MARKERS):
                    group = f"{a}-{b}-{index}-{template}"
                    for active in (0, 1):
                        markers = (own, peer) if active == 0 else (peer, own)
                        text = " ".join(f'{tag}: "{task}".' for tag, task in zip(markers, pair, strict=True))
                        rows.append(dict(group=group, role=roles[active], text=text))
        result[mode] = rows
    if any(len(rows) != 168 or set(Counter(row["role"] for row in rows).values()) != {12}
           for rows in result.values()):
        raise ValueError("Training arms have unequal class counts")
    return dict(seed_sha256=digest(path), datasets=result)


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, generate(args.training))


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.replay_role_training_transfer import checked_banks, evaluate
    from experiments.credit.role_language_router import NISO_SHA256
    from experiments.credit.role_prototypes import normalized
    from experiments.credit.train_role_decision import encode, ridge
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO examples")
    supervised = json.loads(args.generated.read_bytes())
    if supervised != generate(args.training):
        raise ValueError("Generated payload does not match the declared construction")
    glossary = json.loads(args.examples.read_bytes())
    banks = checked_banks(args, glossary)
    holdout = {normalized(text) for bank in banks.values() for variants in bank.values() for text in variants}
    holdout.update(normalized(e["text"]) for e in glossary["examples"])
    audit = json.loads(args.audit.read_bytes())
    transfer = json.loads((args.transfer / "manifest.json").read_bytes())
    groups_path = args.transfer / "pairs.jsonl"
    if audit["fixtures"]["transfer"] != digest(args.transfer / "manifest.json") or digest(groups_path) != transfer["pairs_sha256"]:
        raise ValueError("Changed transfer fixture")
    authors = {s["doi"]: s["roles_by_author"] for s in transfer["sources"] if s["doi"] in audit["allowed"]}
    groups = [g for g in map(json.loads, groups_path.read_text().splitlines()) if g["doi"] in authors]
    if set(authors) != set(audit["allowed"]) or set(authors) != {g["doi"] for g in groups}:
        raise ValueError("Inconsistent transfer source set")

    original = json.loads(args.training.read_bytes())
    fit, validation = [], []
    for r in ROLES:
        texts = sorted(original[r], key=lambda text: stable("fine-tune-split", r, text))
        fit.extend(dict(role=r, text=text) for text in texts[:-1])
        validation.append(dict(role=r, text=texts[-1]))
    train_sets = {"ridge84": [], **supervised["datasets"]}
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    report, training = {}, {}
    with torch.inference_mode():
        started = perf_counter()
        base = torch.cat([encode(encoder, [r["text"] for r in part]) for part in (fit, validation)])
        base_y = torch.tensor([ROLES.index(r["role"]) for r in (*fit, *validation)], device="cuda")
        torch.cuda.synchronize()
        base_seconds = perf_counter() - started
        for name, rows in train_sets.items():
            if holdout.intersection(normalized(r["text"]) for r in rows):
                raise ValueError("Training sentence duplicated in evaluation")
            start = perf_counter()
            x, y = base, base_y
            if rows:
                extra = [encode(encoder, [r["text"] for r in rows[i:i + 32]]) for i in range(0, len(rows), 32)]
                x = torch.cat([base, *extra])
                labels = torch.tensor([ROLES.index(r["role"]) for r in rows], device="cuda")
                y = torch.cat((base_y, labels))
            weights = ridge(x, y)
            torch.cuda.synchronize()
            training[name] = dict(examples=len(y), seconds=base_seconds + perf_counter()-start)
            predictor = lambda x: x @ weights.T
            raw = predictor(encode(encoder, [r["text"] for r in glossary["examples"]])).argmax(1).tolist()
            report[name] = {style: evaluate(encoder, predictor, phrases, groups, authors)
                            for style, phrases in banks.items()}
            report[name]["niso_atomic"] = sum(ROLES[i] == item["role"]
                                               for i, item in zip(raw, glossary["examples"], strict=True))
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 inputs={key: digest(getattr(args, key))
                                         for key in ("audit", "generated", "training", "phrases")},
                                 script_sha256=digest(Path(__file__)), encoder=provenance,
                                 training=training, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("training", type=Path)
    builder.add_argument("output", type=Path)
    test = commands.add_parser("run")
    for name in ("transfer", "audit", "phrases", "examples", "training", "generated", "embedding", "output"):
        test.add_argument(name, type=Path)
    test.add_argument("--comparison-phrases", type=Path)
    args = parser.parse_args()
    (build if args.command == "build" else run)(args)
