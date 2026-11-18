"""Encode the complete LitSearch title-and-abstract index with a frozen checkpoint."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest


def verify_checkpoint(path):
    provenance = json.loads((path / "checkpoint-provenance.json").read_bytes())
    for name, expected in provenance["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Changed embedding artifact: {name}")
    return provenance


def paper_text(row):
    return f"{row['title'] or ''}\n{row['abstract'] or ''}"


def build(projection, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    provenance = verify_checkpoint(checkpoint)
    with gzip.open(projection, "rb") as stream:
        papers = json.load(stream)["papers"]
    texts = [paper_text(row) for row in papers]
    order = sorted(range(len(papers)), key=lambda i: (len(texts[i]), i))
    model = QwenTorchEmbedding(checkpoint, batch_size=64)
    model.model.config.use_cache = False
    start = perf_counter()
    vectors = None
    for offset in range(0, len(order), 2048):
        batch = order[offset:offset + 2048]
        encoded = model.encode([texts[i] for i in batch])
        if vectors is None:
            vectors = np.empty((len(order), encoded.shape[1]), dtype=np.float32)
        vectors[batch] = encoded
        print("Encoded", offset + len(batch), "/", len(order), flush=True)
    offline_seconds = perf_counter() - start
    with output.open("xb") as stream:
        np.savez(stream, corpus_ids=[p["corpusid"] for p in papers], vectors=vectors,
                 projection_sha256=digest(projection), provenance=json.dumps(provenance, sort_keys=True),
                 offline_seconds=offline_seconds, gpu=model.torch.cuda.get_device_name())
    print("Offline embedding seconds", offline_seconds, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.projection, args.checkpoint, args.output)
