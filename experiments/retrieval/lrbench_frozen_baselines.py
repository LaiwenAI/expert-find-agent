"""Compare fixed-profile lexical and dense scoring on the released LR-Bench pairs."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.array_bm25 import ArrayBM25
from expert_search.fast_retrieval import compile_index
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import index
from expert_search.statistics import clustered_paired_interval


ASSET_HASHES = {
    "evaluations_pc.json": "07c9bb8bc4e64b35277a632ed84f2604d8d84ade4eee7da89a9e6e86eba9942a",
    "evaluations_rc.json": "39d54eade077be62aa03553172aa30cdcc15fd63a2c5d2cabb95f34959109d54",
    "RATE_pc.json": "351f1487d314ee7732730d85d129cab8dd2935369c904666b512a9983fbfe3c5",
    "RATE_rc.json": "320eae613fd1fe0682939cb16a3111e0df6af6f9c13133691cdb43de4a643d1e",
    "rate-evaluation_script.py": "ffe9aa9a8f6a4c67cc6f0843439499d6b04b2585926130d5154ea47101f2b85d",
}
ARMS = ("bm25-max", "dense-max", "dense-top3")


def text(paper):
    title = paper.get("paper_title", paper.get("title"))
    return f"{title.strip()}\n{paper['abstract'].strip()}"


def read_split(root, split):
    examples = json.loads((root / f"evaluations_{split}.json").read_bytes())
    upstream = json.loads((root / f"RATE_{split}.json").read_bytes())
    paper_centric = split == "pc"
    anchor_key, other_key = ("paper_id", "author_id") if paper_centric else ("author_id", "paper_id")
    cases = []
    for pair, replay in zip(examples, upstream, strict=True):
        anchor, positive, negative = (pair[key] for key in ("anchor", "positive", "negative"))
        expected = (anchor[anchor_key], positive[other_key], negative[other_key],
                    pair["type"], positive["score"], negative["score"])
        actual = tuple(replay[key] for key in ("anchor_id", "positive_id", "negative_id",
                                              "type", "positive_ref_score", "negative_ref_score"))
        if expected != actual or not positive["score"] > negative["score"]:
            raise ValueError(f"Official predictions do not align with {split} pair {len(cases)}")
        comparisons = [(anchor, positive), (anchor, negative)] if paper_centric else [
            (positive, anchor), (negative, anchor)]
        cases.append(tuple((text(paper), tuple(dict.fromkeys(map(text, reviewer["papers"]))))
                           for paper, reviewer in comparisons))
    return cases, upstream


def build_lexical(doc_ids, root):
    with (root / "evidence.jsonl").open("w") as handle:
        for document, position in doc_ids.items():
            handle.write(json.dumps({"id": str(position), "expert_id": "profile", "text": document}) + "\n")
    index(root, root / "postings.sqlite")
    compile_index(root / "postings.sqlite", root / "compiled.sqlite")
    return ArrayBM25(root / "compiled.sqlite")


def paired_stats(evaluator, split, predictions):
    rate = predictions["rate"]
    groups = [row["anchor_id"] for row in rate]
    correct = {}
    result = {}
    for arm, rows in predictions.items():
        nested = [
            {
                "positive": {"score": row["positive_ref_score"], "pred_score": row["positive_score"]},
                "negative": {"score": row["negative_ref_score"], "pred_score": row["negative_score"]},
            }
            for row in rows
        ]
        accuracy = evaluator.compute_pairwise_accuracy(nested)
        loss = evaluator.compute_pairwise_weighted_kendall_loss(nested)
        correct[arm] = np.array([row["positive_score"] > row["negative_score"] for row in rows],
                                dtype=np.int8)
        if accuracy["correct"] != int(correct[arm].sum()) or accuracy["total"] != len(rows):
            raise ValueError("Unexpected reference-label orientation")
        result[arm] = {"accuracy": accuracy, "weighted_kendall_loss": loss,
                       "ties": sum(row["positive_score"] == row["negative_score"] for row in rows)}
    for arm in ARMS:
        interval = clustered_paired_interval(correct[arm] - correct["rate"], groups)
        interval["sampling_unit"] = "paper anchor" if split == "pc" else "reviewer anchor"
        result[arm]["accuracy_delta_vs_rate"] = interval
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    for name, expected in ASSET_HASHES.items():
        if sha256((args.release / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected upstream asset: {name}")
    spec = importlib.util.spec_from_file_location("rate_evaluator", args.release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    splits = {split: read_split(args.release, split) for split in ("pc", "rc")}
    uses = [arm for cases, _ in splits.values() for pair in cases for arm in pair]
    queries = dict.fromkeys(query for query, _ in uses)
    documents = dict.fromkeys(document for _, profile in uses for document in profile)
    doc_ids = {document: i for i, document in enumerate(documents)}
    query_ids = {query: i for i, query in enumerate(queries)}

    start = perf_counter()
    embedder = QwenTorchEmbedding(args.model, max_length=512, batch_size=32)
    embedder.model.config.use_cache = False
    embedder.encode([next(iter(queries))], query=True)
    initialized = perf_counter()
    doc_vectors = embedder.encode(list(documents))
    indexed = perf_counter()
    query_vectors = embedder.encode(list(queries), query=True)
    encoded = perf_counter()

    with TemporaryDirectory() as directory:
        lexical = build_lexical(doc_ids, Path(directory))
    lexical_indexed = perf_counter()
    lexical_scores = {query: lexical.document_scores(query) for query in queries}
    lexical_scored = perf_counter()

    scores = {}
    for query, profile in dict.fromkeys(uses):
        if not profile:
            raise ValueError("Empty released profile")
        indices = [doc_ids[doc] for doc in profile]
        dense = np.sort(doc_vectors[indices] @ query_vectors[query_ids[query]])
        scores[query, profile] = (
            max(lexical_scores[query].get(str(i), 0.0) for i in indices),
            float(dense[-1]),
            float(dense[-3:].mean()),
        )
    scored = perf_counter()
    summary = {
        "gpu": embedder.torch.cuda.get_device_name(),
        "unique_target_texts": len(queries),
        "unique_profile_texts": len(documents),
        "unique_target_profile_scores": len(scores),
        "asset_sha256": ASSET_HASHES,
        "checkpoint_provenance": json.loads((args.model / "checkpoint-provenance.json").read_bytes()),
        "component_wall_seconds": {
            "dense_initialize_and_warm_up": initialized - start,
            "dense_profile_embedding": indexed - initialized,
            "dense_batched_query_embedding": encoded - indexed,
            "lexical_index_build": lexical_indexed - encoded,
            "lexical_batched_query_scoring": lexical_scored - lexical_indexed,
            "cached_pair_scoring_and_aggregation": scored - lexical_scored,
        },
        "splits": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for split, (cases, upstream) in splits.items():
        predictions = {"rate": upstream}
        for col, arm in enumerate(ARMS):
            rows = []
            for (positive, negative), reference in zip(cases, upstream, strict=True):
                rows.append(dict(reference, positive_score=scores[positive][col],
                                  negative_score=scores[negative][col]))
            (args.output / f"{arm}_{split}.json").write_text(json.dumps(rows, indent=2) + "\n")
            predictions[arm] = rows
        summary["splits"][split] = {
            "pairs": len(cases),
            "anchors": len({row["anchor_id"] for row in upstream}),
            "pairs_with_exact_target_profile_text_overlap": sum(
                any(query in profile for query, profile in pair) for pair in cases),
            "results": paired_stats(evaluator, split, predictions),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
