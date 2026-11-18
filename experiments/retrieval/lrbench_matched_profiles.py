"""Separate compact reviewer profiles from a matched-size checkpoint change."""

import argparse
import ast
from hashlib import file_digest, sha256
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, paired_stats, read_split


ASSETS = {
    **ASSET_HASHES,
    "keywords.json": "480409a808b06977e2786689d5225a9d1b3856bb21ca63632f701fb650dedfe3",
    "rate-RATE.py": "0c64e9532de5102a4e610b7a236c127cbf3951d776d99be417c248562476ffb8",
}
MODEL_PINS = {
    "frozen": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
               "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    "rate06": ("Gnociew/RATE-0.6B", "3afb29314108e4fdf834e48a7b5b30fa64d9afe7",
               "47ce30e2a8b6d0f78ae62108d993452a9e55e598d027c03a81079fae9a25f0c6"),
}


def verify_model(path, name):
    manifest = json.loads((path / "checkpoint-provenance.json").read_bytes())
    files = manifest["files"]
    actual = (manifest["repository"], manifest["revision"], files["model.safetensors"])
    if actual != MODEL_PINS[name]:
        raise ValueError(f"Wrong checkpoint for {name}")
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        with (path / filename).open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != files[filename]:
                raise ValueError(f"Changed checkpoint file: {path / filename}")
    return manifest


def read_inputs(root):
    for name, expected in ASSETS.items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected upstream asset: {name}")

    # Read only the two string literals. The downloaded inference module is not run.
    source = ast.parse((root / "rate-RATE.py").read_bytes())
    names = {"DEFAULT_AUTHOR_INSTRUCTION", "DEFAULT_PAPER_INSTRUCTION"}
    prompts = {stmt.targets[0].id: ast.literal_eval(stmt.value)
               for stmt in source.body if isinstance(stmt, ast.Assign)
               and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in names}
    keywords = {row.get("paper_title") or row["id"]: row["keywords"]
                for row in json.loads((root / "keywords.json").read_bytes())}

    def paper_input(paper):
        title = paper.get("paper_title") or paper["title"]
        return prompts["DEFAULT_PAPER_INSTRUCTION"] + "\n" + f"{title} {paper['abstract']}".strip()

    def candidate(paper, reviewer):
        words = []
        for history in reviewer["papers"]:
            key = history.get("title") or history.get("paper_title")
            value = keywords.get(key) or keywords.get(history.get("id"))
            if not value:
                raise ValueError("Missing released keyword string")
            words.append(value)
        return (paper_input(paper), prompts["DEFAULT_AUTHOR_INSTRUCTION"] + "\n" + " ".join(words),
                tuple(dict.fromkeys(map(paper_input, reviewer["papers"]))))

    splits = {}
    for split in ("pc", "rc"):
        _, replay = read_split(root, split)
        pairs = []
        for row in json.loads((root / f"evaluations_{split}.json").read_bytes()):
            anchor, pos, neg = (row[key] for key in ("anchor", "positive", "negative"))
            if split == "pc":
                pairs.append((candidate(anchor, pos), candidate(anchor, neg)))
            else:
                pairs.append((candidate(pos, anchor), candidate(neg, anchor)))
        splits[split] = pairs, replay
    return splits


def encode_and_score(checkpoint, cases, queries, profiles, documents):
    start = perf_counter()
    embedder = QwenTorchEmbedding(checkpoint, max_length=2048, batch_size=32)
    embedder.model.config.use_cache = False
    embedder.encode([next(iter(queries))], mean_pool=True)
    initialized = perf_counter()
    lengths = embedder.tokenizer([*queries, *profiles, *documents], return_length=True)["length"]
    if max(lengths) > embedder.max_length:
        raise ValueError(f"Unplanned truncation: maximum input length is {max(lengths)}")
    validated = perf_counter()
    query_vecs = embedder.encode(list(queries), mean_pool=True)
    query_encoded = perf_counter()
    profile_vecs = embedder.encode(list(profiles), mean_pool=True)
    profile_encoded = perf_counter()
    document_vecs = embedder.encode(list(documents), mean_pool=True)
    document_encoded = perf_counter()
    compact = {(paper, profile): float(query_vecs[queries[paper]] @ profile_vecs[profiles[profile]])
               for paper, profile in dict.fromkeys((p, k) for p, k, _ in cases)}
    compact_scored = perf_counter()
    raw = {}
    for paper, history in dict.fromkeys((p, h) for p, _, h in cases):
        scores = document_vecs[[documents[doc] for doc in history]] @ query_vecs[queries[paper]]
        raw[paper, history] = float(np.sort(scores)[-3:].mean())
    raw_scored = perf_counter()
    return compact, raw, {
        "gpu": embedder.torch.cuda.get_device_name(),
        "max_input_tokens": max(lengths),
        "initialization_seconds": initialized - start,
        "batch_query_seconds": query_encoded - validated,
        "batch_compact_profile_seconds": profile_encoded - query_encoded,
        "batch_raw_history_seconds": document_encoded - profile_encoded,
        "compact_scores": len(compact),
        "raw_scores": len(raw),
        "compact_dot_seconds": compact_scored - document_encoded,
        "raw_top3_seconds": raw_scored - compact_scored,
    }


def contrasts(predictions, split):
    pairs = {
        "rate06_minus_frozen_with_keywords": ("rate06-keywords", "frozen-keywords"),
        "rate06_minus_frozen_with_raw": ("rate06-raw-top3", "frozen-raw-top3"),
        "keyword_minus_raw_with_rate06": ("rate06-keywords", "rate06-raw-top3"),
        "keyword_minus_raw_with_frozen": ("frozen-keywords", "frozen-raw-top3"),
    }
    correct = {arm: np.array([row["positive_score"] > row["negative_score"] for row in rows],
                             dtype=np.int8) for arm, rows in predictions.items()}
    groups = [row["anchor_id"] for row in predictions["rate"]]
    results = {}
    for label, (first, second) in pairs.items():
        interval = clustered_paired_interval(correct[first] - correct[second], groups)
        interval["sampling_unit"] = "paper anchor" if split == "pc" else "reviewer anchor"
        results[label] = interval
    return results


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    splits = read_inputs(args.release)
    spec = importlib.util.spec_from_file_location("rate_evaluator", args.release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    cases = list(dict.fromkeys(arm for pairs, _ in splits.values() for pair in pairs for arm in pair))
    queries = {key: idx for idx, key in enumerate(dict.fromkeys(paper for paper, _, _ in cases))}
    profiles = {key: idx for idx, key in enumerate(dict.fromkeys(profile for _, profile, _ in cases))}
    documents = {key: idx for idx, key in enumerate(dict.fromkeys(
        doc for _, _, history in cases for doc in history))}
    metrics = {"assets": ASSETS, "queries": len(queries), "compact_profiles": len(profiles),
               "history_documents": len(documents), "checkpoints": {}, "splits": {}}
    scores = {}
    for name, path in (("frozen", args.frozen), ("rate06", args.rate_model)):
        manifest = verify_model(path, name)
        compact, raw, timings = encode_and_score(path, cases, queries, profiles, documents)
        scores[f"{name}-keywords"] = [compact[paper, profile] for paper, profile, _ in cases]
        scores[f"{name}-raw-top3"] = [raw[paper, history] for paper, _, history in cases]
        metrics["checkpoints"][name] = {"provenance": manifest, "timings": timings}

    case_ids = {case: index for index, case in enumerate(cases)}
    args.output.mkdir(parents=True, exist_ok=False)
    for split, (pairs, released) in splits.items():
        predictions = {"rate": released}
        for arm, values in scores.items():
            rows = [dict(reference, positive_score=values[case_ids[pos]], negative_score=values[case_ids[neg]])
                    for (pos, neg), reference in zip(pairs, released, strict=True)]
            (args.output / f"{arm}_{split}.json").write_text(json.dumps(rows, indent=2) + "\n")
            predictions[arm] = rows
        metrics["splits"][split] = {"results": paired_stats(evaluator, split, predictions),
                                    "contrasts": contrasts(predictions, split)}
    (args.output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--rate-model", type=Path, default=Path("artifacts/models/rate-0.6b"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
