"""Retrieve reviewers and attach owned sources from resident MERIT profile bundles."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import merit_author_retriever as merit
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_hybrid import request, reserved_union, score_reciprocals
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, select, summarize
from experiments.retrieval.cof_live_certified_batching import POOL
from experiments.retrieval.merit_resident_profile_bundles import coverage_centers, encode, settings

MERIT_ARMS = ("merit_one", "merit_multi")
ARMS = ("dense", "mean_max", *MERIT_ARMS, "merit_dense_union")
WIDTH = 3
REGULAR_CONTEXT = 4096
LONG_CONTEXT = 8192


def encode_catalog(model, tokenizer, texts):
    unique = list(dict.fromkeys(texts))
    start = perf_counter()
    lengths = tokenizer(unique, truncation=False, return_length=True)["length"]
    if max(lengths) > min(model.config.max_position_embeddings, LONG_CONTEXT):
        raise ValueError("A bundle exceeds the audited long-context budget")
    regular, long = [], []
    for text, length in zip(unique, lengths, strict=True):
        (regular if length <= REGULAR_CONTEXT else long).append(text)
    census_seconds = perf_counter() - start
    vectors, summary = merit.embed_8b(
        model, tokenizer, regular, batch_size=8, max_length=REGULAR_CONTEXT, original_pool=False,
    )
    if long:
        extra, cost = merit.embed_8b(
            model, tokenizer, long, batch_size=1, max_length=LONG_CONTEXT, original_pool=False,
        )
        vectors.update(extra)
        for key in ("texts", "seconds", "padded_tokens", "truncated_texts"):
            summary[key] += cost[key]
    if summary["truncated_texts"]:
        raise ValueError("A source bundle exceeded the MERIT input budget")
    summary.update(census_seconds=census_seconds, maximum_tokens=max(lengths), long_texts=len(long))
    return vectors, summary


class ResidentIndex:
    def __init__(self, reference, papers, model, tokenizer, template):
        self.reference = reference
        self.template = template
        self.model = model
        self.tokenizer = tokenizer
        self.compact = ProfileIndex(reference)
        by_mode = {arm: [] for arm in MERIT_ARMS}
        started = perf_counter()
        for owner in reference.owner_ids:
            positions = reference.owners[owner]
            rows = reference.vectors[positions]
            similarities = rows @ rows.T
            centers = coverage_centers(similarities, WIDTH)
            closest = [(center, *sorted(
                (i for i in range(len(rows)) if i != center),
                key=lambda i: (-similarities[center, i], i),
            )[:WIDTH-1]) for center in centers]
            multi = closest if len(rows) > WIDTH else [centers]
            for mode, groups in zip(MERIT_ARMS, ([centers], multi), strict=True):
                documents = [tuple(reference.source_ids[positions[i]] for i in group) for group in groups]
                by_mode[mode].append(documents)
        self.selection_seconds = perf_counter() - started
        all_groups = dict.fromkeys(group for groups in by_mode.values() for profile in groups for group in profile)
        rendered = {group: template["build_profile_doc_text"]([papers[doc] for doc in group])
                    for group in all_groups}
        embeddings, self.encoding = encode_catalog(model, tokenizer, rendered.values())
        self.arrays = {}
        self.structure = {}
        for mode, profiles in by_mode.items():
            counts = np.asarray([len(profile) for profile in profiles])
            docs = [group for profile in profiles for group in profile]
            packed = np.stack([embeddings[rendered[group]] for group in docs])
            self.arrays[mode] = torch.as_tensor(packed, device="cuda")
            self.structure[mode] = (
                docs, np.cumsum(counts) - counts, np.repeat(np.arange(len(profiles)), counts),
            )

    def prompted_query(self, query):
        title, _, abstract = query.partition("\n")
        text = self.template["get_paper_text"](dict(paper_title=title, abstract=abstract))
        return self.template["get_qwen3_query_prompt"](self.template["DEFAULT_TASK_PROMPT"], text)

    def merit_scores(self, query, mode):
        vector = encode(self.model, self.tokenizer, [self.prompted_query(query)], LONG_CONTEXT)[0]
        docs, starts, owners = self.structure[mode]
        scores = torch.mv(self.arrays[mode], torch.as_tensor(vector, device="cuda")).cpu().numpy()
        owner_scores = np.maximum.reduceat(scores, starts)
        selected = np.flatnonzero(scores == owner_scores[owners])
        unique, first = np.unique(owners[selected], return_index=True)
        if len(unique) != len(owner_scores):
            raise ValueError("Each reviewer must have a winning owned evidence bundle")
        return owner_scores, [docs[i] for i in selected[first]]

    def search_query(self, query, embedding, arm):
        if arm in ("dense", "mean_max"):
            return self.compact.search_query(query, embedding, arm)

        mode = "merit_multi" if arm == "merit_dense_union" else arm
        rank, documents = self.merit_scores(query, mode)
        ref = self.reference
        if arm == "merit_dense_union":
            vector = embedding.encode([query], query=True)[0]
            source_scores = torch.mv(self.compact.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
            dense = np.maximum.reduceat(source_scores[ref.owner_rows], ref.owner_starts)
            combined = score_reciprocals(rank) + score_reciprocals(dense)
            order = np.argsort(-combined, kind="stable")
            selected = reserved_union(order, rank, dense)
            source_views = []
            for i in selected:
                own = ref.owners[ref.owner_ids[i]]
                dense_docs = [ref.source_ids[j] for j in sorted(
                    own, key=lambda j: (-float(source_scores[j]), ref.source_ids[j]),
                )[:WIDTH]]
                source_views.append(list(dict.fromkeys([documents[i][0], *dense_docs, *documents[i]]))[:WIDTH])
            return [(ref.owner_ids[i], docs) for i, docs in zip(selected, source_views, strict=True)]
        order = np.argsort(-rank, kind="stable")[:POOL]
        return [(ref.owner_ids[i], list(documents[i])) for i in order]


def run(args):
    from peft import PeftModel

    settings(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, template = merit.formatting(args.upstream)
    provenance = verify_checkpoint(args.compact)
    root, papers, _, reference, eligible, _, labels = read_data(args.dataset, provenance)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    tested = select(args.dataset, eligible, texts)
    expected = {"KDD": (122, 737, 99594), "SciRepEval": (59, 665, 73257)}[args.dataset]
    if (len(tested), len(reference.owner_ids), len(reference.source_ids)) != expected:
        raise ValueError("Changed full-catalog evaluation panel")
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone, padding_side="right", local_files_only=True, trust_remote_code=False,
    )
    base = AutoModel.from_pretrained(
        args.backbone, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
    ).to("cuda").eval()
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    index = ResidentIndex(reference, papers, adapted, tokenizer, template)
    embedding = QwenTorchEmbedding(args.compact, batch_size=1)
    reranker = Reranker(args.reranker)
    warm = next(pid for pid in eligible if pid not in tested)
    query_lengths = tokenizer([index.prompted_query(texts[qid]) for qid in (warm, *tested)],
                              truncation=False, return_length=True)["length"]
    if max(query_lengths) > LONG_CONTEXT:
        raise ValueError("A query exceeds the audited long-context budget")
    print(f"Built {index.encoding['texts']} bundles, including {index.encoding['long_texts']} long views.", flush=True)
    for arm in ARMS:
        request(index, embedding, reranker, texts, texts[warm], labels[warm], arm)
    measurements = []
    for ordinal, qid in enumerate(tested):
        shift = ordinal % len(ARMS)
        measurements.append({
            arm: request(index, embedding, reranker, texts, texts[qid], labels[qid], arm)
            for arm in ARMS[shift:] + ARMS[:shift]
        })
        if (ordinal + 1) % 20 == 0:
            print(f"Evaluated {ordinal + 1} full-catalog queries.", flush=True)

    metrics = summarize(measurements, [labels[q] for q in tested], arms=ARMS)
    for mode in ARMS:
        values = [entry[mode] for entry in measurements]
        metrics[mode]["positive_and_cited_p5_delta_vs_dense"] = paired_interval([
            r["cited_p5"] - row["dense"]["cited_p5"] for row, r in zip(measurements, values, strict=True)
        ])
        metrics[mode]["top5_exact_order_vs_dense"] = sum(
            [person for person, _ in r["ranked_sources"][:5]] ==
            [person for person, _ in row["dense"]["ranked_sources"][:5]]
            for r, row in zip(values, measurements, strict=True)
        )
    write_json(args.output, dict(
        dataset=args.dataset, queries=len(tested), reviewers=len(reference.owner_ids), sources=len(reference.source_ids),
        positives=sum(sum(v > 0 for v in labels[q].values()) for q in tested),
        maximum_query_tokens=max(query_lengths),
        cohort_sha256=sha256("\n".join(tested).encode()).hexdigest(),
        labels_sha256=digest(root / f"{args.dataset}_queries_test_hard.json"), gpu=torch.cuda.get_device_name(),
        revisions=dict(merit=merit.ADAPTER_REVISION, backbone=merit.BASE_REVISION, embedding=provenance["revision"],
                       reranker=reranker.provenance["revision"]),
        offline=dict(selection_seconds=index.selection_seconds, encoding=index.encoding,
                     vector_bytes={name: matrix.numel() * matrix.element_size() for name, matrix in index.arrays.items()}),
        arms=metrics, script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("upstream", "backbone", "adapter", "compact", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
