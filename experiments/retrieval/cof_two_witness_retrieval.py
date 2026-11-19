"""Cross two-source reviewer ranking with inexpensive evidence selection."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log1p
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from experiments.retrieval import cof_followup_value as follow
from experiments.retrieval.cof_compact_profile_shortlist import STOP, title_tokens


REFERENCE = "9341b1f1da94d4c2681d7ed898d4507e840366f7a1592bb53f6a1eb7db39a0a7"
ARMS = ("raw_dense", "raw_title", "two_dense", "two_title")
SOURCE_WINDOW = 12


class Index(follow.ProfileIndex):
    def __init__(self, reference, papers):
        super().__init__(reference)
        self.owner_positions = np.arange(len(reference.owner_rows))
        self.title_terms = [keywords(papers[doc]["title"]) for doc in reference.source_ids]
        counts = Counter(term for terms in self.title_terms for term in terms)
        self.idf = {term: log1p((len(self.title_terms) - n + .5) / (n + .5))
                    for term, n in counts.items()}

    def ranked(self, vector, use_two):
        ref = self.reference
        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        links = scores[ref.owner_rows]
        counts = self.counts
        first = np.maximum.reduceat(links, ref.owner_starts)
        centroid = self.profile_scores(vector, "mean_max")
        raw = .5 * (centroid + first)
        if use_two:
            is_max = links == np.repeat(first, counts)
            locations = np.minimum.reduceat(np.where(is_max, self.owner_positions, len(links)), ref.owner_starts)
            active = links.copy()
            active[locations] = -np.inf
            second = np.maximum.reduceat(active, ref.owner_starts)
            raw = .50 * centroid + .25 * first + .25 * np.where(counts > 1, second, first)
        return scores, np.argsort(-raw, kind="stable")[:10]

    def sources(self, ranks, scores, query, use_title):
        ref = self.reference
        words = keywords(query) if use_title else frozenset()
        hits = []
        for i in ranks:
            owner = ref.owner_ids[i]
            rows = sorted(ref.owners[owner], key=lambda row: (-float(scores[row]), ref.source_ids[row]))
            selected = rows[:follow.DEPTH]
            if use_title and len(rows) > follow.DEPTH:
                selected = title_fusion(rows[:SOURCE_WINDOW], words, self.title_terms, self.idf)
            hits.append((owner, [ref.source_ids[row] for row in selected]))
        return hits


def keywords(text):
    return frozenset(w for w in title_tokens(text) if len(w) >= 3 and w not in STOP and not w.isdecimal())


def title_fusion(rows, query, title_terms, idf):
    remaining = rows[1:]
    scores = [sum(idf[w] for w in sorted(query & title_terms[row])) for row in remaining]
    lexical = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i))
    ranks = {idx: rank for rank, idx in enumerate(lexical, 1)}

    def fusion(i):
        dense_rank = i + 1
        title_rank = ranks.get(i)
        return 1 / (10 + dense_rank) + (1 / (10 + title_rank) if title_rank is not None else 0.)

    chosen = sorted(range(len(remaining)), key=lambda i: (-fusion(i), i))[:follow.DEPTH - 1]
    return [rows[0], *(remaining[i] for i in chosen)]


def dispatch(request, budget):
    observed = request.initial_values()
    if len(observed) > budget:
        raise ValueError("Shared budget cannot accommodate the common first wave")
    started = perf_counter()
    while len(observed) < budget:
        options = request.ordered_options(observed)
        if not options:
            raise ValueError("Scoring limit exceeds unique available sources")
        doc = request.choose("one_hit", observed, options, None)
        observed[doc] = request.margins[doc]
    return observed, (perf_counter() - started) * 1000


def rows_for_query(index, texts, qid, truth, encoder, verifier, order):
    text = texts[qid]
    started = perf_counter()
    vector = encoder.encode([text], query=True)[0]
    encoding_ms = (perf_counter() - started) * 1000
    pools, retrieval = {}, {}
    for arm in order:
        started = perf_counter()
        scores, ranked = index.ranked(vector, arm.startswith("two_"))
        pools[arm] = index.sources(ranked, scores, text, arm.endswith("_title"))
        retrieval[arm] = encoding_ms + (perf_counter() - started) * 1000

    doc_sets = {arm: {doc for _, sources in hits for doc in sources} for arm, hits in pools.items()}
    documents = sorted(set.union(*doc_sets.values()))
    margins = {}
    truncated = 0
    for start in range(0, len(documents), follow.SCORE_BATCH):
        batch = documents[start:start + follow.SCORE_BATCH]
        scores, clipped = verifier.score(text, texts, batch, task=follow.TASK)
        if not all(np.isfinite(scores)):
            raise ValueError("Nonfinite verifier margins")
        margins.update(zip(batch, scores, strict=True))
        truncated += clipped

    requests = {arm: follow.Request(hits, truth, {doc: margins[doc] for doc in doc_sets[arm]})
                for arm, hits in pools.items()}
    read_counts = {arm: min(len(req.sources), len(req.initial) + follow.FOLLOWUP) for arm, req in requests.items()}
    budget = min(read_counts.values())
    aligned = all(len(req.initial) <= budget for req in requests.values())
    baseline, _ = dispatch(requests["raw_dense"], read_counts["raw_dense"])
    baseline_quality = requests["raw_dense"].quality(baseline)
    results = {}
    for arm, req in requests.items():
        expected = {e for e, grade in truth.items() if grade == 1}
        pool_positive = {e for e, _ in req.hits} & expected
        full = sum(e in expected and sum(margins[x] >= 0 for x in docs) >= 2 for e, docs in req.hits)
        if aligned:
            observed, runtime = dispatch(req, budget)
            results[arm] = dict(req.quality(observed), reads=len(observed), full_pool_checks=len(req.sources),
                                positives_in_pool=len(pool_positive), full_supported_positives=full,
                                retrieval_ms=retrieval[arm], dispatch_ms=runtime)
    return results, baseline_quality, truncated, aligned, len({tuple(x) for x in map(sorted, doc_sets.values())})


def score_panel(dataset, encoder, verifier, provenance, expected):
    root, papers, _, reference, ids, _, labels = follow.read_data(dataset, provenance)
    texts = {pid: (row.get("title", "") + "\n" + row.get("abstract", "")).strip() for pid, row in papers.items()}
    if dataset == "SIGIR":
        order = sorted(ids, key=lambda q: sha256(f"cof-followup-v1:{q}".encode()).digest())
        chosen = order[2 * len(order) // 3:]
    elif dataset == "NIPS":
        chosen = ids
    else:
        chosen = follow.select(dataset, ids, texts)
    cohort = sha256("\n".join(chosen).encode()).hexdigest()
    label_hash = follow.digest(root / f"{dataset}_queries_test_hard.json")
    if cohort != expected["qids_sha256"] or label_hash != expected["labels_sha256"]:
        raise ValueError("Changed frozen benchmark inputs")
    index = Index(reference, papers)
    vector = encoder.encode([texts[chosen[0]]], query=True)[0]
    for arm in ARMS:
        scores, ranked = index.ranked(vector, arm.startswith("two_"))
        index.sources(ranked, scores, texts[chosen[0]], arm.endswith("_title"))
    shared, baseline = [], Counter()
    truncated = changed = 0
    for i, qid in enumerate(chosen):
        shift = i % len(ARMS)
        result, reference_row, clipped, aligned, variants = rows_for_query(
            index, texts, qid, labels[qid], encoder, verifier, ARMS[shift:] + ARMS[:shift]
        )
        baseline.update(reference_row)
        truncated += clipped
        changed += variants > 1
        if aligned:
            shared.append(result)
        if (i + 1) % 40 == 0:
            print(dataset, "completed", i + 1, flush=True)
    for key, value in baseline.items():
        if value != expected["report"]["one_hit"][key]:
            # Scoring the union of arm proposals may alter batch-level numerics.
            print(dataset, "union-shape difference", key, value, expected["report"]["one_hit"][key], flush=True)
    report = {}
    for arm in ARMS:
        rows = [row[arm] for row in shared]
        report[arm] = dict(
            {k: sum(row[k] for row in rows) for k in ("positive", "judged", "cited_positive", "cited_negative",
                                                    "cited_unknown", "positives_in_pool", "full_supported_positives")},
            mean_unique_reads=mean(row["reads"] for row in rows),
            mean_pool_checks=mean(row["full_pool_checks"] for row in rows),
            retrieval_p95_ms=float(np.quantile([row["retrieval_ms"] for row in rows], .95)),
            dispatch_p95_ms=float(np.quantile([row["dispatch_ms"] for row in rows], .95)),
            paired_cited_p5=follow.paired_interval(
                [(row[arm]["cited_positive"] - row["raw_dense"]["cited_positive"]) / follow.TOP for row in shared]
            ),
        )
    return dict(queries=len(chosen), aligned=len(shared), changed_source_pools=changed, truncated_pairs=truncated,
                cohort_sha256=cohort, label_sha256=label_hash, control_unaligned_raw=dict(baseline), report=report)


def run(embedding, reranker, reference, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or follow.digest(reference) != REFERENCE):
        raise ValueError("Require a fresh output and pinned reference on Slurm CUDA")
    provenance = follow.verify_checkpoint(embedding)
    old = json.loads(reference.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = follow.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = follow.Reranker(reranker)
    if old["embedding_revision"] != provenance["revision"] or old["reranker_revision"] != verifier.provenance["revision"]:
        raise ValueError("Changed model checkpoints")
    results = {dataset: score_panel(dataset, encoder, verifier, provenance, old["report"][dataset])
               for dataset in (*follow.DATASETS, "NIPS")}
    follow.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reference_sha256=REFERENCE,
        script_sha256=follow.digest(Path(__file__)), embedding_revision=provenance["revision"],
        reranker_revision=verifier.provenance["revision"], source_window=SOURCE_WINDOW, report=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("embedding", "reranker", "reference", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
