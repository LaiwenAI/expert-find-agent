"""Freeze the existing MuSiQue training recipe without loading transfer datasets."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from musique_policy import Index, SEED, digest_ids, fit, training
from musique_search import ARCHIVE_SHA, QwenTorchEmbedding, key, sha256_file


def main(archive, checkpoint, output):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    with archive.open("rb") as handle:
        if sha256_file(handle) != ARCHIVE_SHA:
            raise ValueError("Changed upstream archive")
    rng = np.random.default_rng(SEED)
    with ZipFile(archive) as bundle, bundle.open("data/musique_ans_v1.0_train.jsonl") as handle:
        rows = list(map(json.loads, handle))
    rows.sort(key=lambda row: key(f"musique-policy-v1:train:{row['id']}"))
    rows = rows[:768]
    signature = digest_ids(rows)
    if signature != "354b9aa85f5defb783e9433fe30f465ff9df330206d7531e3348da9ca7b07563":
        raise ValueError("The original training panel changed")
    encoder = QwenTorchEmbedding(checkpoint, max_length=256, batch_size=64)
    index = Index(rows, encoder)
    training_data = training(rows, index, encoder)
    arrays = {}
    for name, blocks in zip(("learned_static", "learned_adaptive"), training_data, strict=True):
        predict, count = fit(blocks, rng)
        if count != 346063:
            raise ValueError("Training-state pool changed")
        # Keep the original fit routine and export only its numeric closure.
        cells = {name: cell.cell_contents for name, cell in zip(
            predict.__code__.co_freevars, predict.__closure__, strict=True,
        )}
        for field in ("center", "sigma", "w1", "b1", "w2", "b2"):
            arrays[f"{name}_{field}"] = cells[field]
    arrays["training_ids_sha256"] = np.asarray(signature)
    arrays["embedding_revision"] = np.asarray(
        json.loads((checkpoint / "checkpoint-provenance.json").read_text())["revision"]
    )
    np.savez(output, **arrays)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "checkpoint", "output"):
        cli.add_argument(arg, type=Path)
    main(**vars(cli.parse_args()))
