"""Test source selection diversity in a fixed reviewer candidate ranking."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import cof_global_profile_pooling as profiles
from experiments.retrieval.cof_compact_profile_shortlist import read_data, title_tokens
from experiments.retrieval.cof_global_hybrid import request

ARMS = ("raw_prefix", "deduplicate", "diverse")
WINDOW = 12
SOURCES = 3
NEAR_COSINE = .94
DIVERSITY_WEIGHT = .35


def select_sources(similarity, titles, arm):
    count = len(titles)
    if count <= SOURCES or arm == "raw_prefix":
        return list(range(min(count, SOURCES)))
    selected = [0]
    remaining = list(range(1, count))
    if arm == "deduplicate":
        for next_source in remaining:
            if all(titles[next_source] != titles[old] and similarity[next_source, old] < NEAR_COSINE
                   for old in selected):
                selected.append(next_source)
                if len(selected) == SOURCES:
                    break
        # Keep the verification depth aligned when a profile lacks distinct sources.
        selected = (selected + [i for i in remaining if i not in selected])[:SOURCES]
    else:
        while len(selected) < SOURCES:
            def utility(i):
                relevance = 1 - i / (count - 1)
                redundancy = max(1. if titles[i] == titles[j] else similarity[i, j] for j in selected)
                return relevance - DIVERSITY_WEIGHT * redundancy, -i

            chosen = max(remaining, key=utility)
            selected.append(chosen)
            remaining.remove(chosen)
    return sorted(selected)


class SourceSelection:
    def __init__(self, reference, papers):
        self.index = profiles.ProfileIndex(reference)
        self.vectors = reference.vectors
        self.indices = {doc: i for i, doc in enumerate(reference.source_ids)}
        self.titles = {doc: tuple(title_tokens(papers[doc]["title"])) for doc in reference.source_ids}
        self.changed = 0

    def search_query(self, query, encoder, arm):
        hits = self.index.search_query(query, encoder, "mean_max", depth=WINDOW)
        self.changed = 0
        if arm == "raw_prefix":
            return [(expert, docs[:SOURCES]) for expert, docs in hits]
        result = []
        for i, (expert, docs) in enumerate(hits):
            # The shared verifier only reads the original Top-5.
            if i < profiles.TOP:
                vectors = self.vectors[[self.indices[doc] for doc in docs]]
                positions = select_sources(vectors @ vectors.T, [self.titles[doc] for doc in docs], arm)
                chosen = [docs[pos] for pos in positions]
                self.changed += chosen != docs[:SOURCES]
            else:
                chosen = docs[:SOURCES]
            result.append((expert, chosen))
        return result


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = SourceSelection(reference, papers)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = profiles.select(dataset, ids, texts)

    def evaluate(qid, arm):
        truth = labels[qid]
        row = request(index, encoder, verifier, texts, texts[qid], truth, arm)
        return row | dict(
            changed_selection=index.changed,
            judged_negative_with_citations=sum(expert in truth and truth[expert] == 0 and bool(docs)
                                              for expert, docs in row["returned_citations"]),
        )

    warmup = next(qid for qid in ids if qid not in chosen)
    for arm in ARMS:
        evaluate(warmup, arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(ARMS)
        arms = {arm: evaluate(qid, arm) for arm in ARMS[offset:] + ARMS[:offset]}
        rankings = [[expert for expert, _ in arms[arm]["ranked_sources"]] for arm in ARMS]
        if any(ranks != rankings[0] for ranks in rankings[1:]):
            raise AssertionError("Expert ranking changed")
        rows.append(arms)
        if (i + 1) % 30 == 0:
            print(f"{dataset} completed {i + 1} of {len(chosen)} queries", flush=True)

    report = {}
    for arm in ARMS:
        values = [row[arm] for row in rows]
        report[arm] = dict(
            known_positive_p5=mean(x["p5"] for x in values),
            known_positive_cited_p5=mean(x["cited_p5"] for x in values),
            cited_delta=paired_interval([row[arm]["cited_p5"] - row["raw_prefix"]["cited_p5"] for row in rows]),
            citation_coverage=mean(x["citation_coverage"] for x in values),
            changed_selection=sum(x["changed_selection"] for x in values),
            judged_negative_with_citations=sum(x["judged_negative_with_citations"] for x in values),
            complete_p50_p95_ms=np.quantile([x["complete_ms"] for x in values], [.5, .95]).tolist(),
            retrieval_p50_p95_ms=np.quantile([x["retrieval_ms"] for x in values], [.5, .95]).tolist(),
        )
    write_json(output, dict(
        dataset=dataset, queries=len(chosen), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), request_sha256=digest(Path(request.__code__.co_filename)),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
