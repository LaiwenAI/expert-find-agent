"""Allocate the fixed first verification slots using owned evidence per condition."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.retrieval import enterprise_hybrid_rescue as hybrid
from experiments.retrieval import enterprise_joint_first_round as joint
from experiments.retrieval import enterprise_role_first_query as role_index
from experiments.retrieval.enterprise_project_role_rescue import terms
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("hybrid_prefix", "hybrid_joint", "condition_prefix", "condition_joint")
KEEP = 2


def condition_ranks(index, vectors, candidates):
    """Rank each candidate by its strongest owned source for every condition."""
    reference = index.reference
    positions = {expert: n for n, expert in enumerate(candidates)}
    source_positions = sorted(row for name in candidates for row in reference.owners[name])
    owner_of_row = {row: positions[name] for name in candidates for row in reference.owners[name]}
    torch = index.torch
    with torch.inference_mode():
        q = torch.as_tensor(vectors, device=index.device, dtype=torch.float32)
        selected = torch.as_tensor(source_positions, device=index.device, dtype=torch.long)
        values = (index.vectors.index_select(0, selected) @ q.T).cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite condition score")
    maxima = np.full((len(candidates), len(vectors)), -np.inf)
    np.maximum.at(maxima, [owner_of_row[row] for row in source_positions], values)
    order = np.argsort(-maxima, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(len(candidates))[:, None], axis=0)
    # Smaller worst rank preserves balance across requirements.
    return {name: (int(max(ranks[i])), int(sum(ranks[i])), i) for name, i in positions.items()}


def nominate(full, source_plans, original, ranks):
    selected = original[:KEEP]
    used = {item["id"] for item in selected}
    alternatives = sorted(full[KEEP:], key=lambda item: ranks[item["id"]])
    for incumbent in original[KEEP:]:
        count = len(source_plans[incumbent["id"]])
        hit = next((item for item in alternatives if item["id"] not in used
                    and len(source_plans[item["id"]]) >= count), None)
        if hit is None:
            raise ValueError("Cannot retain the baseline per-slot excerpt budget")
        selected.append(hit)
        used.add(hit["id"])
    return selected


def summarize(observations):
    series = {}
    for arm in ARMS:
        rates = {}
        for cohort in ("all", "multi"):
            subset = [record for record in observations if cohort == "all" or record["conditions"] > 1]
            rows = [record["arms"][arm] for record in subset]
            baseline = [record["arms"]["hybrid_prefix"] for record in subset]
            rates[cohort] = {
                "requests": len(rows), "provisional_requests": sum(bool(r["issued"]) for r in rows),
                "provisional_people": sum(len(r["issued"]) for r in rows),
                "changed_shortlist": sum(r["shortlist"] != b["shortlist"] for r, b in zip(rows, baseline, strict=True)),
                "changed_issued_views": sum(r["issued"] != b["issued"] for r, b in zip(rows, baseline, strict=True)),
                "pair_checks": sum(r["pairs"] for r in rows), "clipped": sum(r["clipped"] for r in rows),
                "post_parse_p95_ms": float(np.quantile([r["ms"] for r in rows], .95)),
            }
        series[arm] = rates
    return series


def run(args):
    import torch

    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import (
        CRITERION_CHARS, EMBED_TASK, K, SOURCE_CHARS, SOURCE_DEPTH, verify_candidates,
    )
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.sepef_first_query import lexical_index

    role_index.require_cuda(args.output, args.trace)
    anchor = json.loads(args.anchor.read_bytes())
    if (digest(args.queries) != DEVELOPMENT or digest(args.corpus) != SOURCE_INVENTORY
            or digest(args.prior_trace) != anchor["trace_sha256"]):
        raise ValueError("Unexpected development inputs")
    prior = {row["id"]: row["arms"]["hybrid_prefixes"]
             for row in json.loads(args.prior_trace.read_bytes())["observations"]}
    requests = rows(args.queries)
    if len(requests) != len(prior) or {row["id"] for row in requests} != prior.keys():
        raise ValueError("Frozen qualification frame has changed")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {row["id"]: row for row in rows(args.corpus)}
    ids = list(sources)
    owners = [row["expert_id"] for row in sources.values()]
    embedding_path = args.models / "qwen3-embedding-0.6b"
    revision = verify_checkpoint(embedding_path)["revision"]
    cache = json.loads(args.vector_meta.read_bytes())
    if tuple(cache[key] for key in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")) != (
            SOURCE_INVENTORY, revision, digest(args.vectors), len(ids)):
        raise ValueError("Development vector cache changed")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(np.load(args.vectors, allow_pickle=False),
                                                     ids, [[owner] for owner in owners]))
    windows, _ = role_index.load_role_index(args.role_windows, SOURCE_INVENTORY, sources)
    owned = joint.prepare_owned(windows)
    embedding = QwenTorchEmbedding(embedding_path)
    verifier = Reranker(args.models / "qwen3-reranker-0.6b")
    if (revision != anchor["models"]["embedding"]
            or verifier.provenance["revision"] != anchor["models"]["verifier"]):
        raise ValueError("Model checkpoints do not match the prior run")

    with TemporaryDirectory(dir=args.scratch) as directory:
        lexical = lexical_index([row["text"] for row in sources.values()], Path(directory))

        def one(text, criteria, arm):
            start = perf_counter()
            crossed = arm.startswith("condition_") and 1 < len(criteria) and all(
                0 < len(criterion) <= CRITERION_CHARS for criterion in criteria
            )
            vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
            full = hybrid.hybrid_hits(
                text, vector, index, lexical, owners, ids, hybrid.DEPTH if crossed else K, SOURCE_DEPTH,
            )
            original = full[:K]
            evidence = {item["id"]: joint.prefixes(item, sources, SOURCE_CHARS) for item in full}
            selected = original
            if crossed:
                vectors = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {c}" for c in criteria])
                ranks = condition_ranks(index, vectors, [item["id"] for item in full])
                selected = nominate(full, evidence, original, ranks)
            plans = []
            criterion_words = tuple(map(terms, criteria)) if arm.endswith("_joint") else ()
            for new, incumbent in zip(selected, original, strict=True):
                prefix = evidence[new["id"]][:len(evidence[incumbent["id"]])]
                plans.append(joint.plan(new, prefix, criterion_words, sources, owned, "joint_windows")
                             if criterion_words else prefix)
            output = joint.verify(selected, plans, sources, criteria, verifier, verify_candidates)
            return dict(output, shortlist=[r["id"] for r in selected], plans=plans,
                        ms=(perf_counter() - start) * 1000)

        for arm in ARMS:
            one("Find an expert in optical inspection.", ("Optical inspection", "Prior film testing"), arm)
        observations = []
        order = ARMS if not args.order_offset else ARMS[::-1]
        for n, item in enumerate(requests):
            previous = prior[item["id"]]
            text = item["text"]
            criteria = (text.strip().splitlines()[0][:CRITERION_CHARS],
                        *(text[a:b] for a, b in previous["qualifications"]))
            shift = n % len(ARMS)
            results = {arm: one(text, criteria, arm) for arm in order[shift:] + order[:shift]}
            control = results["hybrid_prefix"]
            if (control["shortlist"] != previous["shortlist"] or control["pairs"] != previous["pairs"]
                    or json.dumps(control["issued"]) != json.dumps(previous["issued"])):
                raise ValueError("Original development control did not reproduce")
            if any(row["pairs"] != control["pairs"] for row in results.values()):
                raise ValueError("Verification budget changed")
            observations.append(dict(id=item["id"], conditions=len(criteria), arms=results))
            if (n + 1) % 20 == 0:
                print(f"Scored {n + 1} paired development requests.", flush=True)
    write(args.trace, {"observations": observations})
    write(args.output, dict(
        arms=summarize(observations), scope="Model provisional outputs under frozen development qualifications.",
        query_sha256=DEVELOPMENT, source_sha256=SOURCE_INVENTORY, trace_sha256=digest(args.trace),
        script_sha256=digest(Path(__file__)), order_offset=args.order_offset, gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "role_windows", "vectors", "vector_meta", "models",
                 "scratch", "anchor", "prior_trace", "output", "trace"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    run(cli.parse_args())
