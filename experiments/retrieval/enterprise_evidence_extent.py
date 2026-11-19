"""Localize missed enterprise evidence by adding sources or reading deeper spans."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_first_query import (
    CRITERION_CHARS, CUTOFF, EMBED_TASK, K, SOURCE_CHARS, GuardedReader, ROLES, read_rows, spans,
)
from experiments.retrieval.enterprise_gap_verification import (
    DEVELOPMENT, SOURCE_INVENTORY, ranked, score, tasks,
)


DEPTH = 8
STRIDE = 400
PACKET_N = 24
ADDITIONS = ("deeper_sources", "later_spans", "interaction")


def order_sources(reference, index, vector, base):
    with torch.inference_mode():
        values = torch.mv(index.vectors, torch.as_tensor(vector, device=index.device, dtype=torch.float32)).cpu().numpy()
    result = []
    for hit in base:
        owner = hit["id"]
        rows = sorted(reference.owners[owner],
                      key=lambda i: (-float(values[i]), reference.source_ids[i]))[:DEPTH]
        ids = [reference.source_ids[i] for i in rows]
        if ids[:len(hit["evidence"])] != [row["id"] for row in hit["evidence"]]:
            raise AssertionError("Changed source prefix")
        result.append((owner, ids))
    return result


def later_span(text, query_terms):
    if len(text) <= SOURCE_CHARS:
        return None
    limit = len(text) - SOURCE_CHARS
    starts = sorted(set(range(STRIDE, limit + 1, STRIDE)) | {limit})
    def priority(start):
        terms = set(tokens(text[start:start + SOURCE_CHARS]))
        return len(terms & query_terms)
    start = max(starts, key=priority)
    return start, text[start:start + SOURCE_CHARS]


def build_extensions(order, sources, criteria, baseline):
    groups = {name: [] for name in ADDITIONS}
    for owner, ids in order:
        for i, criterion in enumerate(criteria):
            if len(criterion) > CRITERION_CHARS or (owner, i) in baseline:
                continue
            term_set = set(tokens(criterion))
            for depth, source_id in enumerate(ids):
                row = sources[source_id]
                if row["expert_id"] != owner:
                    raise AssertionError("Source is not owned by candidate")
                text = row["text"]
                if depth >= 2:
                    excerpt = text[:SOURCE_CHARS]
                    if excerpt:
                        groups["deeper_sources"].append((owner, i, criterion, excerpt, (source_id, 0, len(excerpt))))
                alternate = later_span(text, term_set)
                if alternate is not None:
                    start, excerpt = alternate
                    name = "later_spans" if depth < 2 else "interaction"
                    groups[name].append((owner, i, criterion, excerpt, (source_id, start, start + len(excerpt))))
    return groups


def measure(base, sources, order, criteria, verifier):
    first, second = tasks(base, sources, criteria)
    init = {}
    for pairs in (first, second):
        supported, _, clipped = score(pairs, verifier)
        if clipped:
            raise ValueError("The reference batch contains clipped comparisons")
        for key, citation in supported.items():
            init.setdefault(key, citation)
    groups = build_extensions(order, sources, criteria, init)
    additions = {}
    for name, group in groups.items():
        evidence, _, clipped = score(group, verifier)
        if clipped:
            raise ValueError("An extended source batch contains clipped comparisons")
        additions[name] = evidence
    owners = [owner for owner, _ in order]
    cases = {
        "two_openings": init,
        "eight_openings": additions["deeper_sources"] | init,
        "two_with_later_spans": additions["later_spans"] | init,
        "eight_with_later_spans": additions["interaction"] | additions["later_spans"] | additions["deeper_sources"] | init,
    }
    supported = {name: {row[0] for row in ranked(owners, len(criteria), claims) if row[1]}
                 for name, claims in cases.items()}
    return dict(criterion_slots=len(criteria) * len(owners),
                baseline_pairs=len(first) + len(second),
                pair_counts={key: len(group) for key, group in groups.items()},
                supported_slots={name: len(claims) for name, claims in cases.items()},
                supported_experts={name: len(result) for name, result in supported.items()},
                new_complete_experts=len(supported["eight_with_later_spans"] - supported["two_openings"])), init, additions, groups


def tie(key):
    return sha256(("enterprise-extent-review-v1:" + "|".join(map(str, key))).encode()).hexdigest()


def review_candidates(request, criteria, result):
    _, _, additions, groups = result
    proposed = defaultdict(list)
    observed = {key: citation for group in ADDITIONS for key, citation in additions[group].items()}
    for group in ADDITIONS:
        for owner, i, _, excerpt, citation in groups[group]:
            category = "added_support" if observed.get((owner, i)) == citation else "no_support"
            if category == "no_support" and (owner, i) in observed:
                continue
            overlap = len(set(tokens(criteria[i])) & set(tokens(excerpt)))
            proposed[category].append((overlap, group, owner, i, citation, excerpt))
    rows = []
    for category, choices in proposed.items():
        # At most one candidate view per category per request limits repetition.
        _, group, owner, i, citation, excerpt = min(
            choices, key=lambda row: (-row[0], tie((request["id"], row[2], row[3], *row[4]))))
        case_id = tie((request["id"], owner, i, *citation))[:24]
        rows.append((category, case_id, dict(case_id=case_id, demand=request["text"], criterion=criteria[i],
                    source_excerpt=excerpt, support=None, rationale=None, reviewer_id=None),
                    dict(case_id=case_id, request=request["id"], candidate_owner=owner,
                         criterion_index=i, category=category, group=group, citation=citation)))
    return rows


def write_private(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate(rows):
    fields = ("pair_counts", "supported_slots", "supported_experts")
    result = {name: sum(row[name] for row in rows)
              for name in ("criterion_slots", "baseline_pairs", "new_complete_experts")}
    for field in fields:
        result[field] = {key: sum(row[field][key] for row in rows) for key in rows[0][field]}
    result["queries_with_support"] = {arm: sum(row["supported_experts"][arm] > 0 for row in rows)
                                      for arm in rows[0]["supported_experts"]}
    return result


def run(queries, corpus, models, output, review_dir):
    if output.exists() or review_dir.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new destinations within a Slurm GPU allocation")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed development inputs")
    requests, records = read_rows(queries), read_rows(corpus)
    sources = {row["id"]: row for row in records}
    torch.backends.cuda.matmul.allow_tf32 = False
    embedding_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(embedding_path)
    embedding = QwenTorchEmbedding(embedding_path)
    vectors = embedding.encode([row["text"] for row in records])
    reference = ExactOwnedVectorIndex(vectors, list(sources), [[row["expert_id"]] for row in records])
    index = GpuOwnedVectorIndex(reference)
    reader = GuardedReader(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    rows, review = [], []
    for item in requests:
        text = item["text"]
        clauses = spans(text)
        parsed = reader.evaluate(text, clauses, "qualification_recheck", cutoff=CUTOFF)
        criteria = [text.strip().splitlines()[0][:CRITERION_CHARS]]
        criteria.extend(text[span["start"]:span["end"]] for span, label
                        in zip(clauses, parsed["labels"], strict=True) if ROLES[label] == "prior_qualification")
        if len(criteria) < 2:
            continue
        vector = embedding.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
        base = index.search_vector(vector, k=K)["hits"]
        order = order_sources(reference, index, vector, base)
        result = measure(base, sources, order, criteria, verifier)
        rows.append(result[0])
        review.extend(review_candidates(item, criteria, result))
        if len(rows) % 5 == 0:
            print(f"Probed {len(rows)} multi-criterion requests", flush=True)
    if len(rows) != 25:
        raise AssertionError("Multi-criterion cohort changed")
    counts = {status: sum(row[0] == status for row in review) for status in ("added_support", "no_support")}
    chosen = []
    for status in counts:
        chosen.extend(sorted((r for r in review if r[0] == status), key=lambda r: r[1])[:PACKET_N // 2])
    chosen.sort(key=lambda r: r[1])
    review_dir.mkdir(mode=0o700)
    write_private(review_dir / "blind.jsonl", [row[2] for row in chosen])
    write_private(review_dir / "mapping.jsonl", [row[3] for row in chosen])
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), requests=len(rows),
        corpus_sha256=SOURCE_INVENTORY, development_sha256=DEVELOPMENT, script_sha256=digest(Path(__file__)),
        model_revisions={"embedding": provenance["revision"], "role_reader": reader.provenance["revision"],
                         "verifier": verifier.provenance["revision"]},
        report=aggregate(rows), packet_size=len(chosen),
        packet_categories={status: sum(r[0] == status for r in chosen) for status in counts},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "models", "output", "review-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(**vars(parser.parse_args()))
