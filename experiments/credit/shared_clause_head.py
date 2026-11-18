"""Learn an ownership selector that shares role-normalization embeddings."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.compositional_edits import stable
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.replay_role_training_transfer import checked_banks, evaluate_roles
from experiments.credit.role_clause_selection import clauses, load_groups, reverse_semicolon
from experiments.credit.role_language_router import NISO_SHA256
from experiments.credit.role_prototypes import normalized
from experiments.credit.train_role_decision import encode, ridge
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


FRAMES = (
    ("The designated contributor's work was {}", "The work of a different contributor was {}"),
    ("The article credited the person under review with {}", "The article credited a teammate with {}"),
    ("This person had the responsibility to {}", "A collaborator had the responsibility to {}"),
    ("The duty assigned to the queried investigator was {}", "The duty assigned to a separate investigator was {}"),
    ("What the author in question actually did was {}", "What others actually did was {}"),
    ("The authorship record lists this person's activity as {}", "Another person's activity was recorded as {}"),
)
METHODS = ("raw_ridge", "role_margin", "shared_head")


def seed_rows(raw):
    return [(role, text) for role in ROLES for text in raw[role]]


def paired_fragments(rows):
    return [
        (own.format(text.rstrip(". ")), peer.format(text.rstrip(". ")))
        for _, text in rows for own, peer in FRAMES
    ]


def fit(encoder, training):
    ordered = {
        role: sorted(training[role], key=lambda text: stable("fine-tune-split", role, text))
        for role in ROLES
    }
    primary = seed_rows({role: ordered[role][:-1] for role in ROLES})
    tail = seed_rows({role: ordered[role][-1:] for role in ROLES})
    rows = primary + tail
    y = torch.tensor([ROLES.index(role) for role, _ in rows], device="cuda")
    base_vectors = torch.cat([encode(encoder, [text for _, text in part]) for part in (primary, tail)])
    role_weights = ridge(base_vectors, y)
    comparisons = paired_fragments(rows)
    sentences = [text for pair in comparisons for text in pair]
    values = torch.cat([encode(encoder, sentences[i:i+32]) for i in range(0, len(sentences), 32)])
    delta = values[::2] - values[1::2]
    # Constrain the two fragments to receive opposite preferences.
    pair_weights = delta.T @ torch.linalg.solve(
        delta @ delta.T + .1 * torch.eye(len(delta), device=delta.device),
        torch.ones(len(delta), device=delta.device),
    )
    if not torch.isfinite(pair_weights).all() or not pair_weights.count_nonzero():
        raise ValueError("Unusable fitted selector")
    return role_weights, pair_weights, dict(
        seed_descriptions=len(rows), ownership_pairs=len(comparisons),
        selector_parameters=pair_weights.numel(),
        pair_fit_correct=int(((delta @ pair_weights) > 0).sum()),
    )


def predict(encoder, role_weights, pair_weights, texts, method):
    pool = [[text] if method == "raw_ridge" else clauses(text) for text in texts]
    flattened = [part for pair in pool for part in pair]
    vectors = encode(encoder, flattened)
    role_scores = vectors @ role_weights.T
    if method == "role_margin":
        peaks = role_scores.topk(2, dim=1).values
        ownership = peaks[:, 0] - peaks[:, 1]
    else:
        ownership = vectors @ pair_weights
    roles = role_scores.argmax(dim=1).tolist()
    scores = ownership.tolist()
    result = []
    cursor = 0
    for candidates in pool:
        limit = cursor + len(candidates)
        choice = max(range(cursor, limit), key=lambda i: scores[i])
        result.append(ROLES[roles[choice]])
        cursor = limit
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO examples")
    glossary = json.loads(args.examples.read_bytes())
    phrase_banks = checked_banks(args, glossary)
    training = json.loads(args.training.read_bytes())
    rows = seed_rows(training)
    if len(rows) != 84 or len(set(normalized(text) for _, text in rows)) != 84:
        raise ValueError("Unexpected role training inventory")
    holdout = {normalized(text) for bank in phrase_banks.values() for phrases in bank.values() for text in phrases}
    if holdout.intersection(normalized(text) for pair in paired_fragments(rows) for text in pair):
        raise ValueError("Training fragment duplicates an evaluation phrase")
    groups, authors = load_groups(args.transfer, args.audit)
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    report = {}
    with torch.inference_mode():
        started = perf_counter()
        role, ownership, fit_info = fit(encoder, training)
        torch.cuda.synchronize()
        fit_info["seconds"] = perf_counter() - started

        def resolve(texts, method):
            if args.reverse_semicolon:
                texts = [reverse_semicolon(text) for text in texts]
            return predict(encoder, role, ownership, texts, method)

        for style, phrases in phrase_banks.items():
            report[style] = {}
            probe = [phrases[r][0] for r in ROLES[:4]]
            for method in METHODS:
                resolve(probe, method)
                report[style][method] = evaluate_roles(lambda texts: resolve(texts, method),
                                                       phrases, groups, authors)
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 training_sha256=digest(args.training), script_sha256=digest(Path(__file__)),
                                 phrases_sha256=digest(args.phrases), audit_sha256=digest(args.audit),
                                 reverse_semicolon=args.reverse_semicolon, embedding=provenance,
                                 training=fit_info, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transfer", "audit", "phrases", "examples", "training", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--comparison-phrases", type=Path)
    parser.add_argument("--reverse-semicolon", action="store_true")
    run(parser.parse_args())
