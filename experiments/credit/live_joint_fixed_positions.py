"""Test fixed-width positional encoding for two live topic queries."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding import TASK
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


WIDTH = 64


@torch.inference_mode()
def encode_pair(model, topics, relative):
    if len(topics) != 2:
        raise ValueError("Expected exactly two query slots")
    inputs = [f"Instruct: {TASK}\nQuery:{text}" for text in topics]
    batch = model.tokenizer(inputs, padding="max_length", max_length=WIDTH, return_tensors="pt").to("cuda")
    if batch.input_ids.shape != (2, WIDTH):
        raise ValueError("A query exceeds fixed width")
    if relative:
        batch["position_ids"] = (batch.attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    hidden = model.model(**batch).last_hidden_state[:, -1, :].float()
    return torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()


def evaluate(model, index, subjects, queries, relative):
    topics = sorted({atom[0] for pair in queries for atom in pair})
    reference = {t: encode_pair(model, [t, t], relative)[0] for t in topics}
    count = Counter()
    latency = []
    for atoms in queries:
        pair = [a[0] for a in atoms]
        index.encoder = SimpleNamespace(encode=lambda _, query: encode_pair(model, pair, relative))
        result, elapsed = live.handle(index, None, live.request(atoms), "joint", "structured")
        matches = json.loads(result)["matches"]
        count.update(issued=int(bool(matches)), known=int(bool(matches) and all(
            pair[i] in subjects[matches[0][i + 1]] for i in range(2)
        )))
        latency.append(elapsed)
    vectors = {tuple(a[0] for a in atoms) for atoms in queries}
    comparisons = [float(np.max(np.abs(encode_pair(model, list(pair), relative) - np.stack(
        [reference[t] for t in pair]
    )))) for pair in sorted(vectors)]
    return dict(count, max_abs_vs_same_width_reference=max(comparisons), p95_ms=float(np.quantile(latency, .95)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("elife_xml", "manifest", "embedding", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output on a Slurm compute node")
    checkpoint = live.verify_embedding(args.embedding)
    catalogue, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    subjects, _, queries = live.elife.build_requests(catalogue)["topics"]
    model = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory() as root:
        index = live.LocalIndex(catalogue, model, Path(root) / "elife")
        relative = (False, True) if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) % 2 == 0 else (True, False)
        out = {"relative" if flag else "absolute": evaluate(model, index, subjects, queries, flag)
               for flag in relative}
    write_json(args.output, dict(
        results=out, projection_sha256=checksum, checkpoint=checkpoint, width=WIDTH,
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
