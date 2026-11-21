"""Check trained role heads on frozen engineering phrases before proof assembly."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_role_banks import encode, projections, queries
from experiments.credit.paired_role_prospective_service import FROZEN_SHA
from experiments.credit.paired_role_verification import PairedReader
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_contrast_head import RoleContrastHeads
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    if digest(args.reference) != FROZEN_SHA:
        raise ValueError("Unexpected role reference")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    if digest(args.training) != reference["inputs"]["training"]:
        raise ValueError("Unexpected training input")
    revision = verify_checkpoint(args.embedding)["revision"]
    if revision != reference["embedding_revision"]:
        raise ValueError("Unexpected frozen embedding revision")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    definitions, original, wording = queries(args.niso, args.training)
    banks = projections(encoder, definitions, original)
    reader = PairedReader(args.reader, ROLES)
    if reader.provenance["revision"] != reference["role_reader_revision"]:
        raise ValueError("Unexpected frozen reader revision")
    heads = RoleContrastHeads(args.contrasts, encoder, original, wording)
    cutoff = reference["fit"]["two_batched"]["cutoff"]
    reports = {}
    for view in ("original", "alternate"):
        rows = wording[view]
        totals = Counter()
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            phrases = [phrase for _, _, phrase in batch]
            gold = [ROLES[i] for i, _, _ in batch]
            points = encode(encoder, phrases)
            predictions = heads.predict(points)
            predictions["ridge_top_one"] = [ROLES[i] for i in (points @ banks["ridge"].T).argmax(axis=1)]
            labels, margin, _ = reader.batched(phrases, [ROLES, ROLES])
            predictions["teacher_all_roles"] = labels if margin > cutoff else [None, None]
            for arm, values in predictions.items():
                totals[arm] += sum(got == target for got, target in zip(values, gold, strict=True))
        reports[view] = dict(phrases=len(rows), correct=totals)
    write_json(args.output, dict(reports=reports, training=heads.provenance, gpu=torch.cuda.get_device_name(),
                                embedding=revision, reader=reader.provenance["revision"],
                                slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "reader", "niso", "training", "reference", "contrasts", "output"):
        cli.add_argument(name, type=Path)
    run(cli.parse_args())
