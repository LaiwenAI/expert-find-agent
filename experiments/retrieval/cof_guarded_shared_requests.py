"""Test guarded prefix reuse on mixed expert-retrieval and verification requests."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.qasper.paired_reader import load_reader
from experiments.qasper.adaptive_evidence import digest
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_shared_prefix_reader import LABELS


ARMS = ("batch", "batch_guarded", "shared_guarded")
GUARD_MARGIN = 1.125
REPEATS = 2
TOP_K = 5


def assign_sources(hits):
    """Use the most shared owned source, then fill each remaining expert."""
    unassigned = set(range(len(hits)))
    groups = defaultdict(list)
    for position, hit in enumerate(hits):
        for rank, source in enumerate(hit["evidence"]):
            groups[source["id"]].append((position, rank))
    chosen = {}
    while unassigned:
        def priority(source):
            remaining = [(p, r) for p, r in groups[source] if p in unassigned]
            return -len(remaining), sum(r for _, r in remaining), source

        source = min(groups, key=priority)
        positions = {p for p, _ in groups.pop(source) if p in unassigned}
        if not positions:
            raise ValueError("Expert has no unused owned evidence")
        chosen[source] = sorted(positions)
        unassigned -= positions
    return chosen


def request_prompts(query, hits, papers, ownership, tokenizer):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    q = papers[query]
    common = ("Determine whether the shared source supports this candidate's expertise to review "
              "the submission. Use candidate-specific evidence.\n\nSubmission:\n"
              + q["title"] + "\n" + q["abstract"] + "\n\nShared publication:\n")
    suffixes = [None] * len(hits)
    complete = [None] * len(hits)
    groups = []
    for sid, owners in assign_sources(hits).items():
        source = papers[sid]
        front = encode(common + source["title"] + "\n" + source["abstract"] + "\n\n")
        for i in owners:
            hit = hits[i]
            if sid not in ownership[hit["id"]] or any(w["id"] not in ownership[hit["id"]] for w in hit["evidence"]):
                raise ValueError("Invalid candidate/source association")
            titles = [papers[w["id"]]["title"] for w in hit["evidence"] if w["id"] != sid]
            suffixes[i] = encode("Other documented publications of the candidate:\n" + "\n".join(titles)
                                 + "\n\nA: supports. B: does not support. C: insufficient. Answer:")
            complete[i] = front + suffixes[i]
        groups.append((front, owners))
    return groups, suffixes, complete


def padded(sequences, pad_id, offset=0):
    width = max(map(len, sequences))
    gaps = [width - len(seq) for seq in sequences]
    ids = torch.tensor([[pad_id] * gap + seq for gap, seq in zip(gaps, sequences)],
                       dtype=torch.long, device="cuda")
    mask = torch.arange(width, device="cuda")[None, :] >= torch.tensor(gaps, device="cuda")[:, None]
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0) + offset
    return ids, mask, positions


class ChoiceRunner:
    def __init__(self, reader, tokenizer):
        self.reader = reader
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if any(len(label) != 1 for label in labels):
            raise ValueError("Require one token per choice")
        self.labels = [label[0] for label in labels]
        self.pad_id = tokenizer.pad_token_id

    def full_batch(self, complete):
        ids, mask, positions = padded(complete, self.pad_id)
        out = self.reader(input_ids=ids, attention_mask=mask, position_ids=positions,
                          logits_to_keep=1, use_cache=False)
        return out.logits[:, -1, self.labels]

    def serial_one(self, seq):
        ids = torch.tensor([seq], dtype=torch.long, device="cuda")
        out = self.reader(input_ids=ids, logits_to_keep=1, use_cache=False)
        return out.logits[0, -1, self.labels].float().cpu().numpy()

    def shared_batch(self, prefix, suffixes):
        initial = torch.tensor([prefix], dtype=torch.long, device="cuda")
        cache = self.reader(input_ids=initial, use_cache=True, logits_to_keep=1).past_key_values
        cached_len = len(prefix)
        if cache.get_seq_length() != cached_len:
            raise ValueError("Unexpected prefix length")
        cache.batch_repeat_interleave(len(suffixes))
        ids, mask, positions = padded(suffixes, self.pad_id, cached_len)
        full_mask = torch.cat((torch.ones((len(suffixes), cached_len), dtype=torch.bool, device="cuda"), mask), dim=1)
        out = self.reader(input_ids=ids, attention_mask=full_mask, position_ids=positions,
                          past_key_values=cache, logits_to_keep=1, use_cache=True)
        return out.logits[:, -1, self.labels]

    def decide(self, groups, suffixes, complete, method):
        singleton = [owners[0] for _, owners in groups if len(owners) == 1]
        if method == "shared_guarded" and len(singleton) != len(complete):
            scores = torch.empty((len(complete), len(LABELS)), dtype=torch.float32, device="cuda")
            for front, owners in groups:
                if len(owners) > 1:
                    scores[owners] = self.shared_batch(front, [suffixes[i] for i in owners]).float()
            if singleton:
                scores[singleton] = self.full_batch([complete[i] for i in singleton]).float()
        else:
            scores = self.full_batch(complete).float()
        scores = scores.cpu().numpy()
        fallback = []
        if method != "batch":
            top_two = np.sort(scores, axis=1)[:, -2:]
            fallback = np.flatnonzero(top_two[:, 1] - top_two[:, 0] <= GUARD_MARGIN).tolist()
            for i in fallback:
                scores[i] = self.serial_one(complete[i])
        return scores.argmax(axis=1).tolist(), len(fallback)


def evaluate(dataset, checkpoint, encoder, reader, tokenizer):
    root, papers, source_ids, owned, qids, _, _ = read_data(dataset, checkpoint)
    gpu = GpuOwnedVectorIndex(owned)
    ownership = {expert: {source_ids[i] for i in rows} for expert, rows in owned.owners.items()}
    runner = ChoiceRunner(reader, tokenizer)

    def retrieve(qid):
        text = (papers[qid]["title"] + "\n" + papers[qid]["abstract"]).strip()
        encoded = encoder.encode([text], query=True)[0]
        hits = gpu.search_vector(encoded, k=TOP_K)["hits"]
        if len(hits) != TOP_K:
            raise ValueError("Too few retrieved experts")
        return hits, request_prompts(qid, hits, papers, ownership, tokenizer)

    rows = []
    with torch.inference_mode():
        _, warm_inputs = retrieve(qids[0])
        for arm in ARMS:
            runner.decide(*warm_inputs, arm)
        for j, qid in enumerate(qids):
            hits, inputs = retrieve(qid)
            reference = [int(runner.serial_one(seq).argmax()) for seq in inputs[2]]
            source_groups = assign_sources(hits)
            repeated = sum(len(ids) > 1 for ids in source_groups.values())
            by_arm = defaultdict(list)
            for repeat in range(REPEATS):
                offset = (j + repeat) % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    torch.cuda.synchronize()
                    start = perf_counter()
                    observed, packed = retrieve(qid)
                    choices, checks = runner.decide(*packed, arm)
                    milliseconds = (perf_counter() - start) * 1000
                    if observed != hits:
                        raise ValueError("Candidate or citation order changed between paired requests")
                    by_arm[arm].append(dict(ms=milliseconds, decisions=choices, fallback=checks,
                                            mismatch=sum(a != b for a, b in zip(choices, reference, strict=True))))
            rows.append(dict(query=qid, shared_groups=repeated, sources=len(source_groups), arms=by_arm))
            if len(rows) % 25 == 0:
                print(dataset, "requests", len(rows), "/", len(qids), flush=True)
    return rows, {"source_rows": len(source_ids), "vectors_sha256": digest(root / "qwen-embedding-torch-v1/vectors.npy")}


def summarize(rows):
    if not rows:
        return None
    out = {}
    by_arm = {}
    for arm in ARMS:
        scores = [float(np.median([r["ms"] for r in row["arms"][arm]])) for row in rows]
        by_arm[arm] = scores
        if any(any(run["decisions"] != row["arms"][arm][0]["decisions"] for run in row["arms"][arm])
               for row in rows):
            raise ValueError("Repeated offline choices differ")
        out[arm] = {
            "group_ms_p50_p95": list(map(float, np.quantile(scores, [.5, .95]))),
            "mean_ms": mean(scores),
            "mismatches_vs_serial": sum(r["arms"][arm][0]["mismatch"] for r in rows),
            "serial_fallback_checks": sum(r["arms"][arm][0]["fallback"] for r in rows),
        }
    return {
        "requests": len(rows),
        "expert_checks_per_arm": len(rows) * TOP_K,
        "metrics": out,
        "shared_minus_batch_guarded_ms": paired_interval(
            np.asarray(by_arm["shared_guarded"]) - by_arm["batch_guarded"]),
    }


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = json.loads((args.embedding / "checkpoint-provenance.json").read_bytes())
    for name, expected in checkpoint["files"].items():
        if digest(args.embedding / name) != expected:
            raise ValueError(f"Changed embedder file: {name}")
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    tokenizer, reader, provenance = load_reader(args.reader)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "checkpoint": provenance, "embedding_checkpoint": checkpoint,
        "gpu": torch.cuda.get_device_name(), "guard_margin": GUARD_MARGIN,
        "repeats": REPEATS, "datasets": {},
    }
    for dataset in ("SciRepEval", "KDD"):
        rows, metadata = evaluate(dataset, checkpoint, encoder, reader, tokenizer)
        path = args.output / f"{dataset}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        summary["datasets"][dataset] = dict(
            **metadata, rows_sha256=digest(path),
            mixed=summarize(rows), shared=summarize([r for r in rows if r["shared_groups"]]),
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--reader", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    run(parser.parse_args())
