"""Freeze live depth-12 reviewer-source verdicts for equal-budget routing."""

import argparse
from collections import Counter
from hashlib import sha256
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data, title_tokens
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, select
from experiments.retrieval.cof_live_certified_batching import CAP, TOP, WIDTH, WaveReader


DEPTH = 12
WORDS = re.compile(r"[^\W_]+", re.UNICODE)
FEATURES = ("inverse_rank", "title_jaccard", "query_coverage", "weighted_overlap", "source_length", "shared_owners")


def tokenize(text):
    return {word for word in WORDS.findall(text.casefold()) if len(word) >= 3}


def features(query_title, query, source_title, source, ranks, text_size, idf):
    match = query & source
    title_match = query_title & source_title
    query_mass = sum(idf[word] for word in sorted(query))
    return [
        1 / min(ranks), len(title_match) / max(1, len(query_title | source_title)),
        len(match) / max(1, len(query)),
        sum(idf[word] for word in sorted(match)) / max(1, query_mass),
        math.log1p(text_size) / 8, min(5, len(ranks)) / 5,
    ]


@torch.inference_mode()
def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Use a fresh output on Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(args.embedding)
    root, papers, _, reference, ids, _, labels = read_data(args.dataset, provenance)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    tokens = {pid: tokenize(text) for pid, text in texts.items()}
    titles = {pid: tokenize(p.get("title", "")) for pid, p in papers.items()}
    df = Counter(word for pid in reference.source_ids for word in tokens[pid])
    index = ProfileIndex(reference)
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    verifier = Reranker(args.reranker)
    chosen = select(args.dataset, ids, texts)
    data = []
    for number, qid in enumerate(chosen, 1):
        candidates = index.search_query(texts[qid], encoder, "mean_max", depth=DEPTH)[:TOP]
        sources = {}
        for position, (_, docs) in enumerate(candidates):
            for rank, doc in enumerate(docs, 1):
                sources.setdefault(doc, []).append((position, rank))
        documents = sorted(sources)
        reader = WaveReader(verifier, texts[qid], texts, documents, cache=False)
        reader.maximum = CAP
        verdicts = {}
        for offset in range(0, len(documents), WIDTH):
            chunk = documents[offset:offset+WIDTH]
            values = reader.recheck([("", doc) for doc in chunk])
            verdicts.update(zip(chunk, (margin >= 0 for margin in values), strict=True))
        idf = {word: 1 + math.log((1 + len(reference.source_ids)) / (1 + df[word])) for word in tokens[qid]}
        mapping = {doc: i for i, doc in enumerate(documents)}
        data.append(dict(
            title=sha256(" ".join(title_tokens(papers[qid]["title"])).encode()).hexdigest(),
            candidates=[[mapping[doc] for doc in docs] for _, docs in candidates],
            truth=[labels[qid].get(owner) for owner, _ in candidates],
            features=[features(titles[qid], tokens[qid], titles[doc], tokens[doc], [rank for _, rank in sources[doc]],
                               len(texts[doc]), idf) for doc in documents],
            accepts=[verdicts[doc] for doc in documents],
            token_lengths=[reader.length(doc) for doc in documents],
            truncated=reader.truncated,
        ))
        if number % 20 == 0:
            print(f"{args.dataset}: froze scores for {number}/{len(chosen)} queries", flush=True)
    write_json(args.output, dict(
        dataset=args.dataset, depth=DEPTH, features=FEATURES, data=data, gpu=torch.cuda.get_device_name(),
        members_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        labels_sha256=digest(root / f"{args.dataset}_queries_test_hard.json"),
        vectors_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding=provenance["revision"], reranker=verifier.provenance["revision"], score_batch=WIDTH, token_cap=CAP,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
