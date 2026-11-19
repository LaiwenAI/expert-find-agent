"""Compare lexical, exact dense, and community-routed SE-PEF expert retrieval."""

import argparse
from collections import Counter
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import index
from expert_search.array_bm25 import ArrayBM25
from expert_search.fast_retrieval import compile_index
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.sepef_temporal_panel import SNAPSHOT, TEST_START


ARMS = ("bm25", "dense_global", "dense_community", "centroid_community", "hybrid_community")
TASK = "Find historical answers demonstrating expertise relevant to the community question."
TOP = 10


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


def clean(text):
    parser = PlainText()
    parser.feed(text)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def payload(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    if any(digest(root / name) != digest_value for name, digest_value in manifest["derived_sha256"].items()):
        raise ValueError("Changed time-frozen panel")
    owner_history = json.loads((root / "profiles.json").read_bytes())
    texts = {row["answer_id"]: clean(row["text"])
             for row in map(json.loads, (root / "answers.jsonl").read_bytes().splitlines())}
    owners = sorted(owner_history)
    ids, snippets, positions = [], [], []
    for i, owner in enumerate(owners):
        for timestamp, source in sorted(owner_history[owner], key=lambda item: item[1]):
            if timestamp >= SNAPSHOT:
                raise ValueError("Non-historical answer in the index")
            ids.append(source)
            snippets.append(texts[source])
            positions.append(i)
    if set(ids) != texts.keys() or len(set(ids)) != len(ids):
        raise ValueError("Unexpected ownership or source inventory")
    return owners, ids, snippets, np.array(positions), manifest


class SourceBank:
    def __init__(self, matrix, documents, owner_index):
        self.docs = documents
        self.vectors = matrix[documents].contiguous()
        positions = owner_index[documents]
        self.owners, self.starts = np.unique(positions, return_index=True)
        self.ends = np.r_[self.starts[1:], len(positions)]
        group = torch.as_tensor(np.repeat(np.arange(len(self.owners)), self.ends-self.starts), device="cuda")
        means = torch.zeros((len(self.owners), matrix.shape[1]), device="cuda")
        means.index_add_(0, group, self.vectors)
        self.centroids = torch.nn.functional.normalize(means, dim=1)

    def dense(self, query, centroid=False, k=TOP):
        if centroid:
            scores = (self.centroids @ query).cpu().numpy()
            top = np.argsort(-scores, kind="stable")[:k]
            result = []
            for i in top:
                start, end = self.starts[i], self.ends[i]
                local = int((self.vectors[start:end] @ query).argmax().item())
                result.append((int(self.owners[i]), int(self.docs[start + local])))
            return result
        scores = (self.vectors @ query).cpu().numpy()
        maxima = np.maximum.reduceat(scores, self.starts)
        top = np.argsort(-maxima, kind="stable")[:k]
        result = []
        for i in top:
            start, end = self.starts[i], self.ends[i]
            position = start + int(np.argmax(scores[start:end]))
            result.append((int(self.owners[i]), int(self.docs[position])))
        return result


def lexical_index(texts, stage):
    with (stage / "evidence.jsonl").open("x") as file:
        for i, text in enumerate(texts):
            file.write(json.dumps(dict(id=str(i), expert_id="indexed", text=text)) + "\n")
    index(stage, stage / "source.sqlite")
    compile_index(stage / "source.sqlite", stage / "postings.sqlite")
    return ArrayBM25(stage / "postings.sqlite")


def bm25(lexical, query, owners, allowed=None, k=TOP):
    hits = lexical.document_scores(query)
    best = {}
    for doc, score in hits.items():
        index = int(doc)
        if allowed is not None and index not in allowed:
            continue
        owner = int(owners[index])
        candidate = (-score, index)
        best[owner] = min(best.get(owner, candidate), candidate)
    order = sorted(best, key=lambda o: (best[o][0], o))[:k]
    return [(owner, best[owner][1]) for owner in order]


def fuse(first, second):
    scores, evidence = Counter(), {}
    for order in (first, second):
        for rank, (owner, doc) in enumerate(order, 1):
            scores[owner] += 1 / (60 + rank)
            evidence.setdefault(owner, doc)
    return [(o, evidence[o]) for o in sorted(scores, key=lambda o: (-scores[o], o))[:TOP]]


def measure(row, arm, encoder, lexical, global_bank, community, masks, owner_index, owners, ids):
    started = perf_counter()
    text = clean(row["text"])
    scope = row["community"]
    bank = community.get(scope, global_bank)
    if arm == "bm25":
        result = bm25(lexical, text, owner_index)
    else:
        query = torch.as_tensor(encoder.encode([f"Instruct: {TASK}\nQuery: {text}"])[0], device="cuda")
        if arm == "dense_global":
            result = global_bank.dense(query)
        elif arm == "hybrid_community":
            result = fuse(bank.dense(query, k=50),
                          bm25(lexical, text, owner_index, masks.get(scope), k=50))
        else:
            result = bank.dense(query, centroid=arm == "centroid_community")
    encoded = json.dumps([(owners[o], ids[d]) for o, d in result], separators=(",", ":"))
    ms = (perf_counter() - started) * 1000
    targets = set(row["expert_ids"])
    if any(owner != owner_index[doc] for owner, doc in result):
        raise AssertionError("Returned answer is not owned by the returned expert")
    ranks = [i for i, (owner, _) in enumerate(result, 1) if owners[owner] in targets]
    return dict(ms=ms, hit5=bool(ranks and ranks[0]<=5), mrr10=1/ranks[0] if ranks else 0.,
                response_bytes=len(encoded), target_in_community=any(owners[i] in targets for i in bank.owners))


def summarize(rows):
    return {
        arm: {field: float(np.mean([row[arm][field] for row in rows]))
              for field in ("hit5", "mrr10", "response_bytes", "target_in_community")}
             | {"p95_ms": float(np.quantile([row[arm]["ms"] for row in rows], .95))}
        for arm in ARMS
    }


def run(panel, checkpoint, scratch, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a new output and Slurm CUDA")
    owners, ids, texts, positions, source = payload(panel)
    model = verify_checkpoint(checkpoint)
    if model["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = QwenTorchEmbedding(checkpoint, max_length=256, batch_size=64)
    encoder.model.config.use_cache = False
    encoder.encode(["Warm-up query."])
    start = perf_counter()
    vectors = torch.as_tensor(encoder.encode(texts), device="cuda")
    torch.cuda.synchronize()
    encode_seconds = perf_counter() - start
    global_bank = SourceBank(vectors, np.arange(len(ids)), positions)
    parts = {}
    for i, source_id in enumerate(ids):
        parts.setdefault(source_id.split("_", 1)[0], []).append(i)
    banks = {site: SourceBank(vectors, np.array(indices), positions) for site, indices in parts.items()}
    memberships = {site: set(indices) for site, indices in parts.items()}
    record = {}
    with TemporaryDirectory(dir=scratch, prefix="sepef-retrieval-") as tmp:
        start = perf_counter()
        lexical = lexical_index(texts, Path(tmp))
        lexical_seconds = perf_counter()-start
        for split, boundary in (("val", SNAPSHOT), ("test", TEST_START)):
            panel_rows = json.loads((panel / f"{split}.json").read_bytes())
            if any(row["timestamp"] < boundary for row in panel_rows):
                raise ValueError("Evaluation question precedes its cutoff")
            for arm in ARMS:
                measure(panel_rows[0], arm, encoder, lexical, global_bank, banks, memberships, positions, owners, ids)
            results = []
            for i, row in enumerate(panel_rows):
                offset = i % len(ARMS)
                results.append({arm: measure(row, arm, encoder, lexical, global_bank, banks, memberships,
                                             positions, owners, ids)
                                for arm in ARMS[offset:] + ARMS[:offset]})
            record[split] = dict(queries=len(results), metrics=summarize(results))
            print(split, len(results), "queries", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=model,
        panel_sha256=digest(panel / "manifest.json"), script_sha256=digest(Path(__file__)),
        answers=len(ids), experts=len(owners), communities=len(banks),
        encode_seconds=encode_seconds, lexical_seconds=lexical_seconds, report=record,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "checkpoint", "scratch", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
