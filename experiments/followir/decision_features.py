"""Prepare pinned FollowIR News21 inputs and cache frozen candidate embeddings."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.followir.core17_reranking import digest, prepare


REVISION = "39db677749b3b783bb277d0e2d4712f5f133f52b"
NEWS_FILES = {
    "corpus": "fa09d492d77da79a058ae34abfe30f8b4c0a055552dc069d319fa0a339fa20ee",
    "default": "0d203a3b47b7d28c01e0099f0da3d9e09ca47aa65a1085b2a1ca81175578ad2d",
    "instruction": "b3fb1f3a8cc93b77a341e7140a6839b7ac489f1772c1ad5a5a3b06a0b6eed57e",
    "qrel_diff": "f33ace1a1d1e53e1f34436644e783da4483d3faab770b48269c75429b0ebb2d6",
    "queries": "7c70ebc66d9af14c279bd66f8760ab52391ccd6160f2f1ff30a33ab360dd544a",
    "top_ranked": "8d31c4dde9a586c2947369ee9557bbb20a2ae249fdd68b934bc888fef0af5b3c",
}


def cache(data, checkpoint, output):
    import numpy as np
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding

    dataset = json.loads(data.read_bytes())
    provenance = json.loads((checkpoint / "checkpoint-provenance.json").read_bytes())
    for name, expected in provenance["files"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Changed checkpoint artifact: {name}")
    if output.exists():
        raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = QwenTorchEmbedding(checkpoint)
    model.model.config.use_cache = False
    ids = sorted(dataset["docs"])
    start = perf_counter()
    vectors = model.encode([dataset["docs"][doc] for doc in ids])
    elapsed = perf_counter() - start
    with output.open("xb") as stream:
        np.savez(stream, doc_ids=ids, vectors=vectors, input_sha256=digest(data),
                 checkpoint_provenance=json.dumps(provenance, sort_keys=True), elapsed=elapsed)
    print(len(ids), "candidates encoded in", round(elapsed, 2), "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("parquet_dir", type=Path)
    prep.add_argument("output", type=Path)
    features = subparsers.add_parser("cache")
    for name in ("data", "checkpoint", "output"):
        features.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parquet_dir, args.output, revision=REVISION, files=NEWS_FILES, topics=32, changed_pairs=669)
    else:
        cache(args.data, args.checkpoint, args.output)
