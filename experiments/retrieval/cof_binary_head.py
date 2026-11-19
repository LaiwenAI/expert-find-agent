"""Test a two-token CUDA projection in live CoF citation verification."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.qwen_reranker import pack_tokens
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_live_certified_batching import CAP, GUARD_MARGIN, TASK, TOP, WIDTH, WaveReader


ARMS = ("full", "two_rows", "two_rows_batched_tokenization")


class BinaryWaveReader(WaveReader):
    def __init__(self, reader, query, documents, sources, arm, weight):
        self.arm = arm
        self.weight = weight
        super().__init__(reader, query, documents, sources, dynamic=True, cache=True)

    def _prepare(self, sources):
        if self.arm != "two_rows_batched_tokenization":
            return super()._prepare(sources)
        missing = [source for source in dict.fromkeys(sources) if source not in self.inputs]
        if not missing:
            return
        messages = [
            f"<Instruct>: {TASK}\n<Query>: {self.query}\n<Document>: {self.documents[source]}"
            for source in missing
        ]
        bodies = self.reader.tokenizer(messages, add_special_tokens=False)["input_ids"]
        for source, body in zip(missing, bodies, strict=True):
            tokens, raw = pack_tokens(self.reader.prefix, body, self.reader.suffix, CAP)
            self.inputs[source] = dict(input_ids=tokens)
            self.truncated += raw > len(tokens)
            self.maximum = max(self.maximum, len(tokens))

    @torch.inference_mode()
    def _forward(self, batch, dynamic):
        if not dynamic or self.arm == "full":
            return super()._forward(batch, dynamic)
        tokens = self.reader.tokenizer.pad(batch, padding=True, return_tensors="pt").to("cuda")
        hidden = self.reader.model.model(**tokens, use_cache=False).last_hidden_state[:, -1, :]
        logits = torch.nn.functional.linear(hidden.contiguous(), self.weight).float()
        values = (logits[:, 0] - logits[:, 1]).cpu().tolist()
        self.padded_tokens += len(batch) * tokens["input_ids"].shape[1]
        return values


def request(candidates, reader, weights, texts, query, arm):
    start = perf_counter()
    reader = BinaryWaveReader(reader, query, texts, {doc for _, docs in candidates for doc in docs}, arm, weights)
    output = resolve_shared_frontier(
        candidates, reader.score, reader.recheck, reader.length,
        support=2, top=TOP, width=WIDTH, band=GUARD_MARGIN, cost_aware=False,
    )
    return dict(selected=output["selected"], margins=output["margins"],
                tokens=reader.inputs, rechecks=reader.rechecked_pairs,
                calls=reader.score_calls + reader.recheck_calls, padded=reader.padded_tokens,
                ms=(perf_counter() - start) * 1000)


def summarize(rows):
    result = {}
    for arm in ARMS:
        current = [row[arm] for row in rows]
        prior = [row["full"] for row in rows]
        diffs = [
            a["margins"][key] - b["margins"][key]
            for a, b in zip(current, prior, strict=True) for key in a["margins"].keys() & b["margins"].keys()
        ]
        result[arm] = dict(
            exact_responses=sum(a["selected"] == b["selected"] for a, b in zip(current, prior, strict=True)),
            exact_tokens=sum(a["tokens"] == b["tokens"] for a, b in zip(current, prior, strict=True)),
            exact_margin_maps=sum(a["margins"] == b["margins"] for a, b in zip(current, prior, strict=True)),
            common_max_abs_margin_delta=max(map(abs, diffs)),
            rechecks=sum(row["rechecks"] for row in current),
            mean_forwards=mean(row["calls"] for row in current),
            mean_padded_tokens=mean(row["padded"] for row in current),
            p95_ms=float(np.quantile([row["ms"] for row in current], .95)),
            latency_delta_ms=paired_interval([a["ms"] - b["ms"] for a, b in zip(current, prior, strict=True)]),
        )
    return result


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, _ = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = ProfileIndex(reference)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if reader.model.lm_head.bias is not None:
        raise ValueError("Expected the pinned bias-free Qwen head")
    with torch.inference_mode():
        weights = reader.model.lm_head.weight[reader.labels].contiguous()
    chosen = select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    candidates = index.search_query(texts[warm], encoder, "mean_max")[:TOP]
    for arm in ARMS:
        request(candidates, reader, weights, texts, texts[warm], arm)
    rows = []
    for i, q in enumerate(chosen):
        candidates = index.search_query(texts[q], encoder, "mean_max")[:TOP]
        offset = i % len(ARMS)
        rows.append({arm: request(candidates, reader, weights, texts, texts[q], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 25 == 0:
            print(f"{dataset}: {i + 1}/{len(chosen)}", flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(chosen),
        vocab_rows=reader.model.lm_head.weight.shape[0], output_rows=weights.shape[0],
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(), code_sha256=digest(Path(__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        arms=summarize(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
