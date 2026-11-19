"""Test method-focused query views while keeping evidence verification fixed."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import re
from statistics import mean

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import cof_global_profile_pooling as profiles
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_hybrid import request

ARMS = ("dense", "mean_max", "method_instruction", "lead_sentences", "method_sentences")
TASK = ("Retrieve prior publications that demonstrate the problem-solving techniques, "
        "technical methods or evaluation expertise needed to review the query paper. "
        "Prefer methodological overlap over a shared broad topic.")
BOUNDARY = re.compile(r"(?<=[.!?])\s+")
METHOD = re.compile(
    r"\b(?:propos\w*|introduc\w*|formulat\w*|train\w*|optimiz\w*|optimis\w*|"
    r"algorithm\w*|methodolog\w*|architecture\w*|implement\w*|estimat\w*|"
    r"experiment\w*|benchmark\w*|ablation\w*|simulation\w*|framework\w*)\b",
    re.IGNORECASE,
)


def build_view(query, arm):
    if arm not in {"lead_sentences", "method_sentences"}:
        return query
    title, _, abstract = query.partition("\n")
    sentences = [part.strip() for part in BOUNDARY.split(abstract) if part.strip()]
    indices = list(range(len(sentences)))
    if arm == "method_sentences":
        indices.sort(key=lambda idx: (-len(set(METHOD.findall(sentences[idx].lower()))), idx))
    selected = [sentences[idx] for idx in sorted(indices[:3])]
    return "\n".join([title, " ".join(selected)]).strip()


class InstructionEncoder:
    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, texts, *, query):
        if not query:
            raise ValueError("Method instruction is query-only")
        return self.encoder.encode([f"Instruct: {TASK}\nQuery:{text}" for text in texts])


class SearchViews:
    def __init__(self, index):
        self.index = index
        self.query_characters = 0

    def search_query(self, query, encoder, arm):
        view = build_view(query, arm)
        self.query_characters = len(view)
        if arm == "dense":
            return self.index.search_query(view, encoder, "dense")
        if arm not in ARMS:
            raise ValueError(f"Unexpected arm {arm}")
        use = encoder if arm == "mean_max" else InstructionEncoder(encoder)
        return self.index.search_query(view, use, "mean_max")


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = SearchViews(profiles.ProfileIndex(reference))
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = profiles.select(dataset, ids, texts)

    def evaluate(qid, arm):
        result = request(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
        result["search_query_characters"] = index.query_characters
        return result

    warmup = next(qid for qid in ids if qid not in chosen)
    for arm in ARMS:
        evaluate(warmup, arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: evaluate(qid, arm) for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print(f"{dataset} completed {i + 1} of {len(chosen)} queries", flush=True)

    report = profiles.summarize(rows, [labels[qid] for qid in chosen], arms=ARMS)
    for arm in ARMS:
        report[arm]["search_query_characters"] = mean(row[arm]["search_query_characters"] for row in rows)
    write_json(output, dict(
        dataset=dataset, queries=len(chosen), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), profile_runner_sha256=digest(Path(profiles.__file__)),
        request_sha256=digest(Path(request.__code__.co_filename)),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        known_positives=sum(sum(value > 0 for value in labels[qid].values()) for qid in chosen),
        hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
