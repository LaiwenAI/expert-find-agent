"""Benchmark query-independent expert evidence bundles against live profile selection."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import merit_author_retriever as author

RESIDENT = ("fixed_prefix", "coverage_one", "coverage_multi")
ARMS = ("author_live", *RESIDENT)
WIDTH = 3


def coverage_centers(similarity, limit):
    assigned = np.full(len(similarity), -1.)
    available = np.full(len(similarity), True)
    centers = []
    for _ in range(min(limit, len(similarity))):
        improvements = np.maximum(similarity - assigned, 0.).sum(axis=1)
        center = int(np.argmax(np.where(available, improvements, -np.inf)))
        centers.append(center)
        available[center] = False
        assigned = np.maximum(assigned, similarity[center])
    return tuple(centers)


def evidence_bundles(profile, embedding):
    if not profile:
        raise ValueError("Reviewer has no publications")
    original = tuple(range(min(WIDTH, len(profile))))
    if len(profile) <= WIDTH:
        return {name: (original,) for name in RESIDENT}
    vectors = np.asarray([embedding[author.raw_text(paper)] for paper in profile])
    similarity = vectors @ vectors.T
    centers = coverage_centers(similarity, WIDTH)
    # Each local bundle starts from a different coverage center.
    neighborhoods = tuple((center, *sorted(
        (i for i in range(len(profile)) if i != center), key=lambda i: (-similarity[center, i], i),
    )[:WIDTH-1]) for center in centers)
    return {
        "fixed_prefix": (original,),
        "coverage_one": (centers,),
        "coverage_multi": neighborhoods,
    }


@torch.inference_mode()
def encode(model, tokenizer, texts, max_length, *, original_pool=False):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    positions = inputs.attention_mask.sum(dim=1) - 1
    matrix = model(**inputs, use_cache=False).last_hidden_state[torch.arange(len(texts), device="cuda"), positions]
    return torch.nn.functional.normalize(matrix if original_pool else matrix.float(), dim=1).float().cpu().numpy()


def settings(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Use a fresh output on Slurm CUDA")
    for name, expected in author.WEIGHTS.items():
        if digest(args.backbone / name) != expected:
            raise ValueError("Unexpected backbone checkpoint")
    if digest(args.adapter / "adapter_model.safetensors") != author.ADAPTER_SHA:
        raise ValueError("Unexpected MERIT adapter")


def run(args):
    from peft import PeftModel

    settings(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    pre, template = author.formatting(args.upstream)
    cases = author.load_pairs(args.data, args.rate, "pc", None)
    candidates, queries = {}, {}
    for key, _, comparisons, _ in cases:
        paper, _ = comparisons[0]
        stored_paper, people = queries.setdefault(key, (paper, {}))
        if paper != stored_paper:
            raise ValueError("An anchor ID has conflicting target text")
        for _, candidate in comparisons:
            person = candidate["author_id"]
            publications = candidate["papers"]
            if candidates.setdefault(person, publications) != publications:
                raise ValueError("A reviewer has conflicting profiles")
            people[person] = None

    source_texts = list(dict.fromkeys(author.raw_text(p) for papers in candidates.values() for p in papers))
    compact = QwenTorchEmbedding(args.compact)
    started = perf_counter()
    compact_bank = dict(zip(source_texts, compact.encode(source_texts), strict=True))
    bundle_ids = {person: evidence_bundles(papers, compact_bank) for person, papers in candidates.items()}
    grouping_seconds = perf_counter() - started
    compact_revision = verify_checkpoint(args.compact)["revision"]
    del compact, compact_bank
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone, padding_side="right", local_files_only=True, trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        args.backbone, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
    ).to("cuda").eval()
    render = pre["build_prefilter_paper_text"]
    offline_sources = [render(p) for papers in candidates.values() if len(papers) > WIDTH for p in papers]
    pre_bank, pre_cost = author.embed_8b(model, tokenizer, offline_sources, 8, 2048, original_pool=True)
    adapted = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    profiles = {
        person: {arm: tuple(template["build_profile_doc_text"]([candidates[person][i] for i in bundle])
                            for bundle in groups[arm]) for arm in RESIDENT}
        for person, groups in bundle_ids.items()
    }
    resident, offline_cost = author.embed_8b(
        adapted, tokenizer, [view for group in profiles.values() for views in group.values() for view in views],
        8, 4096, original_pool=False,
    )
    resident_rows = {
        person: {arm: np.asarray([resident[view] for view in group[arm]]) for arm in RESIDENT}
        for person, group in profiles.items()
    }
    print("Built resident profile embeddings and full-history source cache.", flush=True)

    def score(arm, paper, people):
        start = perf_counter()
        query_text = template["get_qwen3_query_prompt"](template["DEFAULT_TASK_PROMPT"],
                                                     template["get_paper_text"](paper))
        if arm == "author_live":
            evidence = [candidates[person] for person in people]
            if any(len(profile) > WIDTH for profile in evidence):
                with adapted.disable_adapter():
                    retrieval = encode(adapted, tokenizer, [render(paper)], 2048, original_pool=True)[0]
                evidence = [
                    sorted(profile, key=lambda p: -float(retrieval @ pre_bank[render(p)]))[:WIDTH]
                    if len(profile) > WIDTH else profile
                    for profile in evidence
                ]
            views = [template["build_profile_doc_text"](profile) for profile in evidence]
            vector, *documents = encode(adapted, tokenizer, [query_text, *views], 4096)
            rankings = {person: float(vector @ document) for person, document in zip(people, documents, strict=True)}
            online_texts = len(views) + 1 + int(any(len(candidates[p]) > WIDTH for p in people))
        else:
            vector = encode(adapted, tokenizer, [query_text], 4096)[0]
            rankings = {person: float(np.max(resident_rows[person][arm] @ vector)) for person in people}
            online_texts = 1
        return rankings, online_texts, (perf_counter() - start) * 1000

    first_paper, first_people = next(iter(queries.values()))
    for arm in ARMS:
        score(arm, first_paper, list(first_people))
    evidence = {arm: {} for arm in ARMS}
    online = {arm: [] for arm in ARMS}
    timing = {arm: [] for arm in ARMS}
    for ordinal, (query_id, (paper, people)) in enumerate(queries.items()):
        rotate = ordinal % len(ARMS)
        for arm in ARMS[rotate:] + ARMS[:rotate]:
            values, count, elapsed = score(arm, paper, list(people))
            evidence[arm][query_id] = values
            online[arm].append(count)
            timing[arm].append(elapsed)
        if (ordinal + 1) % 25 == 0:
            print(f"Completed {ordinal+1} target queries with rotating arm order.", flush=True)
    margins = {"rate": np.asarray([pair[3]["positive_score"] - pair[3]["negative_score"] for pair in cases])}
    for arm in ARMS:
        margins[arm] = np.asarray([
            evidence[arm][key][items[0][1]["author_id"]] - evidence[arm][key][items[1][1]["author_id"]]
            for key, _, items, _ in cases
        ])
    metrics = author.paired_metrics(cases, margins)
    reference = margins["author_live"] > 0
    groups = [row[0] for row in cases]
    rng = np.random.default_rng(20261121)
    draws = rng.integers(len(queries), size=(4000, len(queries)))
    latency_reference = np.quantile(np.asarray(timing["author_live"])[draws], .95, axis=1)
    for arm in ARMS:
        values = np.asarray(timing[arm])
        latency_difference = np.quantile(values[draws], .95, axis=1) - latency_reference
        outcome = margins[arm] > 0
        metrics[arm].update(
            online_text_encodings=sum(online[arm]),
            warm_p95_ms=float(np.quantile(values, .95)),
            warm_p50_ms=float(np.quantile(values, .50)),
            paired_p95_delta_ms_vs_live=np.quantile(latency_difference, [.025, .975]).tolist(),
            paired_accuracy_vs_live=clustered_paired_interval(
                outcome.astype(int) - reference.astype(int), groups,
            ),
            added_correct_pairs_vs_live=int((outcome & ~reference).sum()),
            lost_correct_pairs_vs_live=int((reference & ~outcome).sum()),
        )
        if arm in RESIDENT:
            metrics[arm]["resident_rows"] = sum(len(row[arm]) for row in resident_rows.values())
            metrics[arm]["resident_vector_bytes"] = sum(row[arm].nbytes for row in resident_rows.values())
    metrics["author_live"]["resident_vector_bytes"] = sum(vector.nbytes for vector in pre_bank.values())
    write_json(args.output, dict(
        pairs=len(cases), anchors=len(queries), reviewers=len(candidates), metrics=metrics,
        revisions={"backbone": author.BASE_REVISION, "adapter": author.ADAPTER_REVISION, "compact": compact_revision},
        offline={"coverage_grouping_seconds": grouping_seconds, "author_source_cache": pre_cost,
                 "resident_profiles": offline_cost},
        data_sha256=author.DATA_SHA["pc"], script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "upstream", "rate", "backbone", "adapter", "compact", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
