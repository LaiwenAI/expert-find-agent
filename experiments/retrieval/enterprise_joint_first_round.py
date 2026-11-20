"""Compare enterprise evidence allocation under a fixed first-read budget."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.retrieval import enterprise_hybrid_rescue as hybrid
from experiments.retrieval import enterprise_role_first_query as role_index
from experiments.retrieval.enterprise_project_role_rescue import terms
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("hybrid_prefixes", "independent_windows", "joint_windows")
POOL_DEPTH = 16


def prefixes(hit, sources, max_chars):
    return tuple((entry["id"], 0, min(max_chars, len(sources[entry["id"]]["text"])))
                 for entry in hit["evidence"] if sources[entry["id"]]["text"])


def prepare_owned(windows):
    result = defaultdict(dict)
    for item in windows:
        view = item["view"]
        key = (view["id"], view["start"], view["end"])
        result[item["expert"]][key] = item["terms"]
    return result


def plan(hit, original, criteria, sources, owned, arm):
    if arm == "hybrid_prefixes" or len(criteria) < 2 or len(original) < 2:
        return original
    inventory = dict(owned.get(hit["id"], {}))
    for key in original:
        source, start, end = key
        inventory.setdefault(key, terms(sources[source]["text"][start:end]))
    scores = {key: tuple(len(c & words) / max(len(c), 1) for c in criteria)
              for key, words in inventory.items()}

    def independent(key):
        score = scores[key]
        return -min(score), -sum(score), -int(key in original), key

    indexed = sorted(inventory, key=independent)[:POOL_DEPTH]
    candidates = list(dict.fromkeys([*indexed, *original]))
    if arm == "independent_windows":
        return tuple(candidates[:len(original)])

    def joint(pair):
        combined = tuple(map(max, zip(*(scores[key] for key in pair), strict=True)))
        return (-min(combined), -sum(combined), -int(pair[0][0] != pair[1][0]),
                -sum(key in original for key in pair), pair)

    return min(combinations(candidates, len(original)), key=joint)


def verify(hits, plans, sources, criteria, verifier, verify_candidates):
    aliases = {}
    views = []
    for hit, planned in zip(hits, plans, strict=True):
        view = []
        for reference in planned:
            source_id, start, end = reference
            original = sources[source_id]
            if original["expert_id"] != hit["id"] or not 0 <= start < end <= len(original["text"]):
                raise ValueError("Selected evidence did not match its owner or source")
            token = len(aliases)
            aliases[token] = dict(expert_id=hit["id"], text=original["text"][start:end], reference=reference)
            view.append({"id": token})
        views.append({"id": hit["id"], "evidence": view})
    decisions, pairs, clipped, withheld = verify_candidates(views, aliases, criteria, verifier)
    results = []
    for expert, accepted, citations in decisions:
        if accepted:
            real = []
            for alias, start, end in citations:
                source, source_start, _ = aliases[alias]["reference"]
                real.append([source, source_start + start, source_start + end])
            results.append(dict(expert=expert, citations=list(dict.fromkeys(tuple(c) for c in real))))
    return dict(issued=results, pairs=pairs, clipped=clipped, withheld=withheld)


def summarize(observations):
    reference = [row["arms"][ARMS[0]] for row in observations]
    rng = np.random.default_rng(5419)
    draws = rng.integers(len(reference), size=(4000, len(reference)))
    p95 = lambda arr: np.quantile(np.asarray(arr)[draws], .95, axis=1)
    base = p95([row["ms"] for row in reference])
    out = {}
    for arm in ARMS:
        tested = [row["arms"][arm] for row in observations]
        ms = [row["ms"] for row in tested]
        out[arm] = dict(
            requests=len(tested), issued_requests=sum(bool(r["issued"]) for r in tested),
            provisional_slots=sum(len(r["issued"]) for r in tested),
            changed_requests=sum(r["issued"] != a["issued"] for r, a in zip(tested, reference, strict=True)),
            changed_selection=sum(r["plan"] != a["plan"] for r, a in zip(tested, reference, strict=True)),
            passages=sum(sum(map(len, r["plan"])) for r in tested),
            distinct_sources=sum(len({v[0] for group in r["plan"] for v in group}) for r in tested),
            verification_pairs=sum(r["pairs"] for r in tested),
            clipped=sum(r["clipped"] for r in tested), withheld=sum(r["withheld"] for r in tested),
            p95_ms=float(np.quantile(ms, .95)), paired_p95_delta_ms=np.quantile(p95(ms) - base, [.025, .975]).tolist(),
            planning_p95_ms=float(np.quantile([r["planning_ms"] for r in tested], .95)),
        )
    return out


def run(args):
    import torch

    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import (
        CRITERION_CHARS, CUTOFF, EMBED_TASK, K, ROLES, SOURCE_CHARS, SOURCE_DEPTH, spans, verify_candidates,
    )
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
    from experiments.retrieval.sepef_first_query import lexical_index

    role_index.require_cuda(args.output, args.trace)
    if digest(args.queries) != DEVELOPMENT or digest(args.corpus) != SOURCE_INVENTORY:
        raise ValueError("Require only the pinned development data")
    torch.backends.cuda.matmul.allow_tf32 = False
    requests = rows(args.queries)
    sources = {row["id"]: row for row in rows(args.corpus)}
    ids = list(sources)
    owners = [row["expert_id"] for row in sources.values()]
    embedding_path = args.models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(embedding_path)["revision"]
    cache = json.loads(args.vector_meta.read_bytes())
    if tuple(cache[key] for key in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")) != (
            SOURCE_INVENTORY, checkpoint, digest(args.vectors), len(sources)):
        raise ValueError("Development vector cache changed")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(np.load(args.vectors, allow_pickle=False),
                                                     ids, [[owner] for owner in owners]))
    windows, _ = role_index.load_role_index(args.role_windows, SOURCE_INVENTORY, sources)
    owned = prepare_owned(windows)
    parser = ComparisonParser(args.models / "qwen3-4b-instruct-2507")
    embedding = QwenTorchEmbedding(embedding_path)
    verifier = Reranker(args.models / "qwen3-reranker-0.6b")

    with TemporaryDirectory(dir=args.scratch) as directory:
        lexical = lexical_index([row["text"] for row in sources.values()], Path(directory))
        def one(text, arm):
            started = perf_counter()
            clauses = spans(text)
            parsed = parser.evaluate(text, clauses, "broadcast_prefix", CUTOFF)
            qualifications = [(clause["start"], clause["end"]) for clause, label in zip(
                clauses, parsed["labels"], strict=True) if ROLES[int(label)] == "prior_qualification"]
            criteria = (text.strip().splitlines()[0][:CRITERION_CHARS], *(text[a:b] for a, b in qualifications))
            vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
            hits = hybrid.hybrid_hits(text, vector, index, lexical, owners, ids, K, SOURCE_DEPTH)
            original = [prefixes(hit, sources, SOURCE_CHARS) for hit in hits]
            plan_start = perf_counter()
            query_terms = tuple(map(terms, criteria)) if arm != ARMS[0] else ()
            selected = [plan(hit, pref, query_terms, sources, owned, arm)
                        for hit, pref in zip(hits, original, strict=True)]
            plan_time = (perf_counter() - plan_start) * 1000
            if list(map(len, original)) != list(map(len, selected)):
                raise ValueError("Changed fixed evidence budget")
            result = verify(hits, selected, sources, criteria, verifier, verify_candidates)
            result.update(plan=selected, shortlist=[hit["id"] for hit in hits], qualifications=qualifications,
                          planning_ms=plan_time, ms=(perf_counter() - started) * 1000)
            return result

        for arm in ARMS:
            one("Find an expert in optical inspection with prior polymer film testing work.", arm)
        observations = []
        for n, item in enumerate(requests):
            shift = (n + args.order_offset) % len(ARMS)
            compared = {arm: one(item["text"], arm) for arm in ARMS[shift:] + ARMS[:shift]}
            ref = compared[ARMS[0]]
            for record in compared.values():
                for key in ("shortlist", "qualifications", "pairs"):
                    if record[key] != ref[key]:
                        raise ValueError("Lost paired comparison invariants")
            observations.append(dict(id=item["id"], arms=compared))
            if (n + 1) % 20 == 0:
                print(f"Measured {n + 1} paired development requests", flush=True)
    write(args.trace, {"observations": observations})
    write(args.output, dict(
        arms=summarize(observations), source_records=len(sources), role_windows=len(windows),
        multi_condition=sum(bool(row["arms"][ARMS[0]]["qualifications"]) for row in observations),
        script_sha256=digest(Path(__file__)), trace_sha256=digest(args.trace), gpu=torch.cuda.get_device_name(),
        order_offset=args.order_offset, queries_sha256=DEVELOPMENT, source_sha256=SOURCE_INVENTORY,
        models={"embedding": checkpoint, "verifier": verifier.provenance["revision"],
                "parser": parser.readers["bf16"].provenance["revision"]},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "role_windows", "vectors", "vector_meta", "models", "scratch", "output", "trace"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    run(parser.parse_args())
