"""Evaluate independent packed sequences in live CoF evidence verification."""

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention.varlen import varlen_attn
from transformers import AttentionInterface

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_binary_head import BinaryWaveReader
from experiments.retrieval.cof_frontier_wave import SUPPORT, WaveScheduler, run, verify
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP


ARMS = ("frontier4", "frontier8", "ragged8")
PACKED_ATTENTION = "cof_independent_packed"


def packed_attention(module, query, key, value, attention_mask, *, offsets, max_length, scaling, **kwargs):
    if attention_mask is not None:
        raise ValueError("The cumulative offsets already delimit each attention sequence")
    def tokens_first(tensor):
        return tensor.squeeze(0).transpose(0, 1).contiguous()

    output = varlen_attn(
        tokens_first(query), tokens_first(key), tokens_first(value),
        offsets, offsets, max_length, max_length, scale=scaling, window_size=(-1, 0),
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.unsqueeze(0), None


class RaggedWaveReader(BinaryWaveReader):
    @torch.inference_mode()
    def _forward(self, batch, dynamic):
        if not dynamic or len(batch) == 1:
            return super()._forward(batch, dynamic)
        sequences = [row["input_ids"] for row in batch]
        lengths = [len(sequence) for sequence in sequences]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        token_ids = torch.tensor([[token for seq in sequences for token in seq]], device="cuda")
        positions = torch.tensor([[pos for length in lengths for pos in range(length)]], device="cuda")
        endpoints = torch.tensor([n - 1 for n in offsets[1:]], device="cuda")
        limits = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        model = self.reader.model.model
        config = model.config
        prior = config._attn_implementation
        config._attn_implementation = PACKED_ATTENTION
        try:
            hidden = model(input_ids=token_ids, position_ids=positions, attention_mask={"full_attention": None},
                           offsets=limits, max_length=max(lengths), use_cache=False).last_hidden_state[0]
        finally:
            config._attn_implementation = prior
        logits = torch.nn.functional.linear(hidden.index_select(0, endpoints), self.weight).float()
        self.padded_tokens += offsets[-1]
        return (logits[:, 0] - logits[:, 1]).cpu().tolist()


def setup(reader, weights):
    if set(reader.model.model.config.layer_types) != {"full_attention"}:
        raise ValueError("Packed scorer currently requires full context causal layers")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    wave = RaggedWaveReader(reader, "", {}, (), "two_rows_batched_tokenization", weights)
    first = reader.tokenizer.encode("A researcher studies a different topic.", add_special_tokens=False)
    second = reader.tokenizer.encode("A researcher shows strong evidence for this study.", add_special_tokens=False)
    original = wave._forward([{"input_ids": first}, {"input_ids": second}], True)
    changed = wave._forward([{"input_ids": list(reversed(first))}, {"input_ids": second}], True)
    if original[1] != changed[1]:
        raise ValueError("Packed attention failed the isolation check")
    return {"packed_attention_isolation": True}


def measure(reader, weights, candidates, texts, query, arm, relevant):
    if arm != "ragged8":
        return verify(reader, weights, candidates, texts, query, arm, relevant)
    started = perf_counter()
    wave = RaggedWaveReader(reader, query, texts, {doc for _, docs in candidates for doc in docs},
                            "two_rows_batched_tokenization", weights)
    policy = WaveScheduler(wave, candidates, 8, False)
    out = resolve_shared_frontier(candidates, policy.score, policy.recheck, wave.length,
                                  support=SUPPORT, top=TOP, width=8, band=GUARD_MARGIN, cost_aware=False)
    return dict(
        selected=out["selected"], ms=(perf_counter() - started) * 1000,
        forwards=wave.score_calls + wave.recheck_calls, docs=wave.scored_documents,
        padded_tokens=wave.padded_tokens, rechecks=wave.rechecked_pairs,
        cited_positives=sum(expert in relevant and bool(sources) for expert, sources in out["selected"]),
    )


def summarize(rows):
    result = {}
    for arm in ARMS:
        views = [row[arm] for row in rows]
        comparisons = {}
        for baseline in ("frontier4", "frontier8"):
            ref = [row[baseline] for row in rows]
            comparisons[baseline] = dict(
                exact_outputs=sum(a["selected"] == b["selected"] for a, b in zip(views, ref, strict=True)),
                latency_delta=paired_interval([a["ms"] - b["ms"] for a, b in zip(views, ref, strict=True)]),
            )
        result[arm] = dict(
            comparisons=comparisons, cited_positives=sum(row["cited_positives"] for row in views),
            p95_ms=float(np.quantile([row["ms"] for row in views], .95)),
            **{key: mean(row[key] for row in views) for key in ("forwards", "docs", "padded_tokens", "rechecks")},
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()), methods=ARMS, measure=measure, aggregate=summarize, setup=setup)
