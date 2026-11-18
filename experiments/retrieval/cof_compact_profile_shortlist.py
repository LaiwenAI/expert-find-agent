"""Evaluate source-linked compact reviewer profiles as a global retrieval route."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding


BUDGETS = (20, 50)
TOP_K = 5
TERM_LIMIT = 32
STOP = frozenset("a an and are as at based by for from in into is of on or the to towards using via with".split())


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def title_tokens(title):
    return re.findall(r"[a-z0-9]+", title.casefold())


def profile_terms(title):
    words = title_tokens(title)
    singles = {word for word in words if len(word) >= 3 and word not in STOP and not word.isdecimal()}
    bigrams = {f"{left} {right}" for left, right in zip(words, words[1:]) if left in singles and right in singles}
    return singles | bigrams


def extract_profiles(papers, pids, index):
    terms = [profile_terms(papers[pid]["title"]) for pid in pids]
    frequency = Counter(term for source in terms for term in source)
    weights = {term: math.log1p((len(pids) - count + .5) / (count + .5))
               for term, count in frequency.items()}
    profiles = []
    for expert in index.owner_ids:
        source_rows = index.owners[expert]
        local_frequency = Counter(term for row in source_rows for term in terms[row])
        shortlist = sorted(local_frequency, key=lambda term: (
            -math.log1p(local_frequency[term]) * weights[term], term))[:TERM_LIMIT]
        if not shortlist:
            raise ValueError(f"No extractive terms for {expert}")
        profiles.append((expert, "; ".join(shortlist)))
    return profiles


def read_data(dataset, checkpoint):
    root, papers, reviewers, labels = load(dataset)
    folder = root / "qwen-embedding-torch-v1"
    protocol = json.loads((folder / "protocol.json").read_bytes())
    manifest = json.loads((folder / "summary.json").read_bytes())
    vector_path = folder / "vectors.npy"
    if protocol["checkpoint"] != checkpoint or digest(vector_path) != manifest["vectors_sha256"]:
        raise ValueError("Cached vector provenance mismatch")
    ownership = {}
    for expert, profile in reviewers.items():
        for pid in set(profile["papers"]):
            ownership.setdefault(pid, []).append(expert)
    pids, qids = sorted(ownership), sorted(labels["hard"])

    def paper_text(pid):
        return (papers[pid].get("title", "") + "\n" + papers[pid].get("abstract", "")).strip()

    items = [dict(id="doc:" + p, text=paper_text(p), query=False) for p in pids]
    items += [dict(id="query:" + q, text=paper_text(q), query=True) for q in qids]
    if sha256(json.dumps(items, sort_keys=True).encode()).hexdigest() != protocol["input_sha256"]:
        raise ValueError("Cached source or query order mismatch")
    vectors = np.load(vector_path, allow_pickle=False)
    if vectors.shape != (len(items), 1024):
        raise ValueError("Unexpected cached vector shape")
    titles = {tuple(title_tokens(papers[p]["title"])) for p in pids}
    exact_text = {paper_text(p).casefold() for p in pids}
    tested = [i for i, q in enumerate(qids) if q not in ownership and
              tuple(title_tokens(papers[q]["title"])) not in titles and paper_text(q).casefold() not in exact_text]
    source_index = ExactOwnedVectorIndex(vectors[:len(pids)], pids, [ownership[p] for p in pids])
    ids = [qids[i] for i in tested]
    return root, papers, pids, source_index, ids, vectors[[len(pids) + i for i in tested]], labels["hard"]


def metric(shortlist, hits, truth, baseline):
    ids = [row["id"] for row in hits]
    present = set(ids)
    known_positive = {eid for eid, grade in truth.items() if grade == 1}
    base = set(baseline)
    gained, lost = present - base, base - present
    awarded = sum(truth.get(e, 0) for e in gained) - sum(truth.get(e, 0) for e in lost)
    return {
        "judged_positions": len(present & truth.keys()),
        "known_positive_hits": len(present & known_positive),
        "known_positive_in_shortlist": len(set(shortlist) & known_positive),
        "known_positives": len(known_positive),
        "exact_top5_in_shortlist": len(set(shortlist) & base),
        "exact_top5_reproduced": ids == baseline,
        "paired_p5_low": (awarded - len(lost - truth.keys())) / TOP_K,
        "paired_p5_high": (awarded + len(gained - truth.keys())) / TOP_K,
    }


def summarize(rows):
    out = {}
    for method in rows[0]["arms"]:
        observed = [row["arms"][method] for row in rows]
        positives = sum(row["known_positives"] for row in observed)
        p50, p95 = np.quantile([t for row in observed for t in row["ms"]], [.50, .95])
        out[method] = {
            "known_positive_recall_micro": sum(row["known_positive_in_shortlist"] for row in observed) / positives,
            "known_positive_p5_lower_bound": mean(row["known_positive_hits"] / TOP_K for row in observed),
            "judged_p5_coverage": mean(row["judged_positions"] / TOP_K for row in observed),
            "exact_top5_shortlist_retention": mean(row["exact_top5_in_shortlist"] / TOP_K for row in observed),
            "queries_with_exact_top5_order": sum(row["exact_top5_reproduced"] for row in observed),
            "paired_p5_delta_bounds": [mean(row[key] for row in observed)
                                      for key in ("paired_p5_low", "paired_p5_high")],
            "mean_source_rows": mean(row["source_rows"] for row in observed),
            "cached_component_ms_p50_p95": [float(p50), float(p95)],
        }
    return out


def evaluate(dataset, model, checkpoint):
    root, papers, pids, reference, qids, queries, truth = read_data(dataset, checkpoint)
    gpu = GpuOwnedVectorIndex(reference)
    start = perf_counter()
    profiles = extract_profiles(papers, pids, reference)
    extracted = perf_counter()
    keyword_matrix = model.encode([profile for _, profile in profiles])
    encoded = perf_counter()
    centers = np.stack([reference.vectors[reference.owners[e]].mean(axis=0) for e in reference.owner_ids])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    matrices = {"keyword": keyword_matrix, "centroid": centers}
    roots = {name: torch.as_tensor(matrix, dtype=torch.float32, device="cuda") for name, matrix in matrices.items()}
    methods = [("full", None, None)] + [
        (f"{name}-{k}", name, k) for name in roots for k in BUDGETS]

    def retrieve(query, name, limit):
        start = perf_counter()
        if name is None:
            candidates = reference.owner_ids
            hits = gpu.search_vector(query, k=TOP_K)
        else:
            with torch.inference_mode():
                q = torch.as_tensor(query, dtype=torch.float32, device="cuda")
                scores = torch.mv(roots[name], q).cpu().numpy()
            positions = np.argsort(-scores, kind="stable")[:limit]
            candidates = [reference.owner_ids[i] for i in positions]
            hits = gpu.search_vector(query, k=TOP_K, eligible=candidates)
        return candidates, hits["hits"], hits["source_rows"], (perf_counter() - start) * 1000

    for _, name, limit in methods:
        retrieve(queries[0], name, limit)
    owned = {e: {pids[i] for i in reference.owners[e]} for e in reference.owner_ids}
    rows = []
    for i, (qid, query) in enumerate(zip(qids, queries, strict=True)):
        runs = {method: [] for method, _, _ in methods}
        for repeat in range(5):
            shift = (i + repeat) % len(methods)
            for arm, name, limit in methods[shift:] + methods[:shift]:
                runs[arm].append(retrieve(query, name, limit))
        baseline = [hit["id"] for hit in runs["full"][0][1]]
        row = {"query_id": qid, "arms": {}}
        for arm, trials in runs.items():
            candidates, hits, source_rows, _ = trials[0]
            if any(other != hits for _, other, _, _ in trials[1:]):
                raise ValueError(f"Inconsistent repeated scoring for {qid} {arm}")
            for hit in hits:
                if not hit["evidence"] or any(source["id"] not in owned[hit["id"]] or
                                              not papers[source["id"]]["title"].strip()
                                              for source in hit["evidence"]):
                    raise ValueError("Source cannot be resolved to an owned original")
            row["arms"][arm] = {
                **metric(candidates, hits, truth[qid], baseline),
                "ms": [trial[3] for trial in trials], "hits": hits, "source_rows": source_rows,
            }
        rows.append(row)
    report = {
        "queries": len(rows), "excluded_overlap_queries": len(truth) - len(rows),
        "reviewer_roots": len(profiles), "source_rows": len(pids), "term_limit": TERM_LIMIT,
        "extract_profile_seconds": extracted - start, "encode_profile_seconds": encoded - extracted,
        "repeats_per_query": 5, "metrics": summarize(rows),
        "vector_sha256": digest(root / "qwen-embedding-torch-v1/vectors.npy"),
        "hard_labels_sha256": digest(root / f"{dataset}_queries_test_hard.json"),
    }
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = json.loads((args.model / "checkpoint-provenance.json").read_bytes())
    for name, checksum in checkpoint["files"].items():
        if digest(args.model / name) != checksum:
            raise ValueError(f"Changed checkpoint: {name}")
    model = QwenTorchEmbedding(args.model)
    model.model.config.use_cache = False
    model.encode(["academic reviewer expertise"])
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"gpu": torch.cuda.get_device_name(), "checkpoint": checkpoint, "datasets": {}}
    for dataset in ("NIPS", "SciRepEval"):
        rows, report = evaluate(dataset, model, checkpoint)
        path = args.output / f"{dataset}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n")
        report["rows_sha256"] = digest(path)
        summary["datasets"][dataset] = report
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
