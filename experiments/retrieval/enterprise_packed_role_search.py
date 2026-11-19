"""Test packed verification in the live enterprise role-index fallback stack."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_reranker import pack_tokens
from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_role_first_query as role_search
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("padded_whole", "padded_rescue", "packed_rescue", "guarded_rescue")
TOKEN_RATIO = .80
SIGN_MARGIN = .125


class Reader:
    def __init__(self, canonical):
        import torch
        from transformers import AttentionInterface
        from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
        from experiments.retrieval.cof_ragged_shared_weights import packed_view

        self.torch = torch
        self.forward = packed_forward
        self.canonical = canonical
        model = canonical.model
        if model.lm_head.bias is not None or set(model.model.config.layer_types) != {"full_attention"}:
            raise ValueError("Expected the pinned full-attention bias-free reader")
        AttentionInterface.register(PACKED_ATTENTION, packed_attention)
        with torch.inference_mode():
            self.backbone = packed_view(model.model, share_weights=True)
            self.weight = model.lm_head.weight[canonical.labels].contiguous()
        if any(a.data_ptr() != b.data_ptr()
               for a, b in zip(self.backbone.parameters(), model.model.parameters(), strict=True)):
            raise ValueError("Packed model must share the original weights")
        self.change("padded_whole")

    def change(self, arm):
        self.arm = arm
        self.counts = Counter()

    def encode(self, pairs, task):
        sequences = []
        truncated = 0
        for query, document in pairs:
            body = self.canonical.tokenizer.encode(
                f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {document}",
                add_special_tokens=False,
            )
            ids, raw = pack_tokens(self.canonical.prefix, body, self.canonical.suffix, 1024)
            sequences.append(dict(input_ids=ids))
            truncated += raw > len(ids)
        return sequences, truncated

    def padded(self, sequences, tokens):
        self.counts.update(padded_batches=1, executed_tokens=tokens)
        inputs = self.canonical.tokenizer.pad(sequences, padding=True, return_tensors="pt").to("cuda")
        logits = self.canonical.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.canonical.labels].float()
        return (logits[:, 0] - logits[:, 1]).cpu().tolist()

    def score_pairs(self, pairs, *, task):
        with self.torch.inference_mode():
            sequences, truncated = self.encode(pairs, task)
            lengths = [len(row["input_ids"]) for row in sequences]
            padded = max(lengths) * len(lengths)
            actual = sum(lengths)
            self.counts.update(logical_pairs=len(pairs), logical_tokens=actual)
            use_packed = len(lengths) > 1 and (
                self.arm == "packed_rescue" or (self.arm == "guarded_rescue" and actual < TOKEN_RATIO * padded)
            )
            if not use_packed:
                return self.padded(sequences, padded), truncated
            margins, tokens = self.forward(self.backbone, self.weight, sequences)
            self.counts.update(packed_batches=1, executed_tokens=tokens)
            if self.arm == "guarded_rescue" and any(abs(margin) <= SIGN_MARGIN for margin in margins):
                self.counts.update(guard_rechecks=1)
                margins = self.padded(sequences, padded)
            return margins, truncated


def one(text, arm, parser, embedder, index, sources, role_index, reader):
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, K, TASK, evaluate

    reader.change(arm)
    started = perf_counter()
    original = evaluate(text, "broadcast_prefix", parser, embedder, index, sources, reader)
    primary_counts = dict(reader.counts)
    initial = [dict(expert=eid, citations=[list(c) for c in dict.fromkeys(citations)])
               for eid, accepted, citations in original["decisions"] if accepted]
    criteria = [text.strip().splitlines()[0][:CRITERION_CHARS],
                *(text[a:b] for a, b in original["qualifications"])]
    reader.counts.clear()
    extra, _, invoked, clipped = role_search.recover(
        criteria, [row[0] for row in original["decisions"]], {item["expert"] for item in initial},
        role_index, reader, "whole" if arm == "padded_whole" else "global_on_abstain", TASK, K - len(initial),
    ) if all(len(c) <= CRITERION_CHARS for c in criteria) else ([], 0, False, 0)
    return dict(
        initial=initial, extra=extra, ranked=original["retrieved"], qualifications=original["qualifications"],
        decisions=original["decisions"], primary=primary_counts, fallback=dict(reader.counts),
        primary_verifier_ms=original["times"]["verify_and_rank"] * 1000,
        clipped=original["clipped_pairs"] + clipped, invoked=invoked, elapsed_ms=(perf_counter() - started) * 1000,
    )


def compare(observations):
    values = {arm: [r["arms"][arm] for r in observations] for arm in ARMS}
    anchor = values["padded_rescue"]
    bootstrap = np.random.default_rng(20261120).integers(len(anchor), size=(6000, len(anchor)))
    base_time = np.asarray([r["elapsed_ms"] for r in anchor])
    report = {}
    for arm in ARMS:
        rows = values[arm]
        elapsed = np.asarray([r["elapsed_ms"] for r in rows])
        paired = np.quantile(elapsed[bootstrap], .95, axis=1) - np.quantile(base_time[bootstrap], .95, axis=1)
        counts = {stage: dict(sum((Counter(row[stage]) for row in rows), Counter())) for stage in ("primary", "fallback")}
        report[arm] = dict(
            **counts, issued=sum(bool(row["initial"] or row["extra"]) for row in rows),
            extra=sum(len(row["extra"]) for row in rows), clipped=sum(row["clipped"] for row in rows),
            p50_p95_ms=np.quantile(elapsed, [.50, .95]).tolist(),
            paired_p95_ms_delta_vs_padded_rescue=np.quantile(paired, [.025, .975]).tolist(),
            primary_verifier_p95_ms=float(np.quantile([r["primary_verifier_ms"] for r in rows], .95)),
            changed={key: sum(row[key] != original[key] for row, original in zip(rows, anchor, strict=True))
                     for key in ("initial", "extra", "ranked", "decisions", "qualifications", "invoked")},
        )
    return report


def run(queries, corpus, roles, vectors, meta, models, output, trace):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import TASK
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    role_search.require_cuda(output, trace)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Use the pinned development request and source inventory")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = review.index(rows(corpus), lambda source: source["id"])
    saved = json.loads(meta.read_bytes())
    encoder_path = models / "qwen3-embedding-0.6b"
    info = verify_checkpoint(encoder_path)
    signature = SOURCE_INVENTORY, info["revision"], digest(vectors), len(sources)
    if signature != tuple(saved[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Source cache provenance differs")
    embedding = QwenTorchEmbedding(encoder_path)
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()],
    ))
    role_index = role_search.load_role_index(roles, SOURCE_INVENTORY, sources)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    canonical = Reranker(models / "qwen3-reranker-0.6b")
    reader = Reader(canonical)
    probes = [("Optical metrology experience", "Built a sensor"), ("Optical metrology experience",
               "A researcher validated an optical measurement system on industrial samples.")]
    expected, cutoff = canonical.score_pairs(probes, task=TASK)
    matched, matched_cutoff = reader.score_pairs(probes, task=TASK)
    if matched != expected or cutoff != matched_cutoff:
        raise ValueError("Padded reader changed the original contract")
    reader.change("packed_rescue")
    base = reader.score_pairs(probes, task=TASK)[0]
    other = reader.score_pairs([("Biomass", "A changed text"), probes[1]], task=TASK)[0]
    if base[1] != other[1]:
        raise ValueError("Packed sequences are not isolated")
    fixture = (parser, embedding, index, sources, role_index, reader)
    for arm in ARMS:
        one("Find an expert in optical inspection with prior polymer testing work.", arm, *fixture)
    observations = []
    for i, request in enumerate(rows(queries)):
        shift = i % len(ARMS)
        arms = {arm: one(request["text"], arm, *fixture) for arm in ARMS[shift:] + ARMS[:shift]}
        if any(arms[arm]["ranked"] != arms["padded_whole"]["ranked"] or
               arms[arm]["qualifications"] != arms["padded_whole"]["qualifications"] for arm in ARMS):
            raise ValueError("Unrelated retrieval or parsing change")
        observations.append(dict(id=request["id"], arms=arms))
        if (i + 1) % 20 == 0:
            print(f"Measured {i+1} development queries", flush=True)
    fingerprint = dict(
        code_sha256=digest(Path(__file__)), vectors_sha256=saved["vectors_sha256"], queries_sha256=DEVELOPMENT,
        role_windows_sha256=digest(roles), models=dict(encoder=info["revision"],
            parser=parser.readers["bf16"].provenance["revision"], verifier=canonical.provenance["revision"]),
    )
    write(trace, dict(observations=observations, **fingerprint))
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), **fingerprint,
        requests=len(observations), token_ratio=TOKEN_RATIO, sign_margin=SIGN_MARGIN, metrics=compare(observations),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries", "corpus", "roles", "vectors", "meta", "models", "output", "trace"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
