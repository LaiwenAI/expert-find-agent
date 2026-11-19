"""Test a one-slot expert substitution with evidence from competing profiles."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.retrieval import cof_global_profile_pooling as base


ARMS = ("blend", "global_specificity", "local_sources", "local_rank", "local_joint")
TOP = 5
DEPTH = 3
RELEVANCE_SLACK = .03
ASSOCIATION_PENALTY = .025
SUPPORT_MARGIN = 0.
TASK = "Find prior research whose methods or subject support expertise to review the target paper."


class CompetingProfiles(base.ProfileIndex):
    def __init__(self, reference):
        super().__init__(reference)
        self.owners_of = [set() for _ in reference.source_ids]
        for owner, sources in reference.owners.items():
            for source in sources:
                self.owners_of[source].add(owner)
        self.penalty = ASSOCIATION_PENALTY * np.log2([len(owners) for owners in self.owners_of])

    def shortlist(self, query, encoder, arm):
        vector = encoder.encode([query], query=True)[0]
        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        ref = self.reference
        anchoring = self.centroids @ vector
        maxima = np.maximum.reduceat(scores[ref.owner_rows], ref.owner_starts)
        full_scores = .5 * (anchoring + maxima)
        shortlist = np.argsort(-full_scores, kind="stable")[:base.POOL]
        owner_ids = [ref.owner_ids[i] for i in shortlist]
        protected = set(owner_ids[:TOP - 1])
        incumbent = owner_ids[TOP - 1]
        sources_per_slot = min(DEPTH, len(ref.owners[incumbent]))
        local = arm in ("local_rank", "local_joint", "local_sources")
        local_weights = {}

        def source_weights(owner):
            if owner not in local_weights:
                other = protected - {owner}
                local_weights[owner] = {
                    source: (scores[source] - ASSOCIATION_PENALTY * len(other & self.owners_of[source]))
                    for source in ref.owners[owner]
                }
            return local_weights[owner]

        if arm in ("global_specificity", "local_rank", "local_joint"):
            normalized = {}
            for i, owner in zip(shortlist[TOP - 1:], owner_ids[TOP - 1:], strict=True):
                if len(ref.owners[owner]) < sources_per_slot or full_scores[i] < full_scores[shortlist[TOP - 1]] - RELEVANCE_SLACK:
                    continue
                witness = max(source_weights(owner).values()) if local else max(
                    scores[src] - self.penalty[src] for src in ref.owners[owner]
                )
                normalized[owner] = .5 * (anchoring[i] + witness)
            winner = max(normalized, key=normalized.get)
            if winner != incumbent and normalized[winner] > normalized[incumbent]:
                i = owner_ids.index(winner)
                owner_ids[TOP - 1], owner_ids[i] = owner_ids[i], owner_ids[TOP - 1]

        local_citations = arm in ("local_sources", "local_joint")
        hits = []
        for slot, owner in enumerate(owner_ids[:TOP]):
            values = source_weights(owner) if local_citations else scores
            depth = sources_per_slot if slot == TOP - 1 else min(DEPTH, len(ref.owners[owner]))
            sources = sorted(ref.owners[owner], key=lambda src: (-float(values[src]), ref.source_ids[src]))[:depth]
            hits.append((owner, [ref.source_ids[src] for src in sources]))
        return hits, owner_ids[TOP - 1] != incumbent, owner_ids


def request(index, encoder, verifier, texts, query, labels, arm):
    started = perf_counter()
    hits, replaced, shortlist = index.shortlist(query, encoder, arm)
    retrieved = perf_counter()
    prompts = [(query, texts[source]) for _, sources in hits for source in sources]
    margins, truncated = verifier.score_pairs(prompts, task=TASK)
    elapsed = perf_counter() - started
    if not np.isfinite(margins).all() or len(margins) != len(prompts):
        raise ValueError("Invalid single-batch judgments")
    positives = {expert for expert, value in labels.items() if value > 0}
    known = {expert for expert, _ in hits}
    citations = []
    offset = 0
    for owner, sources in hits:
        supported = [source for source, score in zip(sources, margins[offset:offset + len(sources)], strict=True)
                     if score > SUPPORT_MARGIN]
        if len(supported) >= 2:
            citations.append((owner, supported[:2]))
        offset += len(sources)
    repeated = Counter(src for _, sources in citations for src in sources)
    return dict(ids=known, positive=len(known & positives), negatives=len((known & labels.keys()) - positives),
                positive_at_10=len(set(shortlist) & positives), cited_positive=sum(e in positives for e, _ in citations),
                cited_negative=sum(e in labels and e not in positives for e, _ in citations),
                cited_unjudged=sum(e not in labels for e, _ in citations),
                judged=len(known & labels.keys()), coverage=len(citations),
                repeated=sum(n - 1 for n in repeated.values()), replaced=int(replaced),
                pairs=len(prompts), truncated=truncated, retrieval_ms=(retrieved - started) * 1000, ms=elapsed * 1000)


def summarize(rows, labels):
    table = {}
    for arm in ARMS:
        values = [r[arm] for r in rows]
        bounds = []
        for row, truth in zip(rows, labels, strict=True):
            trial, baseline = row[arm], row["blend"]
            if trial["pairs"] != baseline["pairs"] or trial["positive_at_10"] != baseline["positive_at_10"]:
                raise ValueError("Candidate universe or pair budget changed")
            gained, lost = trial["ids"] - baseline["ids"], baseline["ids"] - trial["ids"]
            delta = trial["positive"] - baseline["positive"]
            bounds.append(((delta - len(lost - truth.keys())) / TOP, (delta + len(gained - truth.keys())) / TOP))
        table[arm] = {
            **{field: sum(v[field] for v in values) for field in (
                "positive", "positive_at_10", "negatives", "cited_positive", "cited_negative", "cited_unjudged",
                "judged", "coverage", "repeated", "replaced", "pairs", "truncated")},
            **{f"p95_{field}": float(np.quantile([v[field] for v in values], .95))
               for field in ("retrieval_ms", "ms")},
            "paired_p5_delta": paired_interval([(r[arm]["positive"] - r["blend"]["positive"]) / TOP for r in rows]),
            "paired_cited_p5_delta": paired_interval(
                [(r[arm]["cited_positive"] - r["blend"]["cited_positive"]) / TOP for r in rows]),
            "unknown_label_p5_delta_bounds": [mean(bound[0] for bound in bounds), mean(bound[1] for bound in bounds)],
        }
    return table


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = base.verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = base.read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = CompetingProfiles(reference)
    encoder = base.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = base.Reranker(reranker)
    chosen = ids if dataset == "SIGIR" else base.select(dataset, ids, texts)
    for arm in ARMS:
        request(index, encoder, verifier, texts, texts[chosen[0]], labels[chosen[0]], arm)
    rows = []
    for i, q in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: request(index, encoder, verifier, texts, texts[q], labels[q], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print(f"{dataset}: completed {i+1}/{len(chosen)}", flush=True)
    base.write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(chosen), experts=len(reference.owner_ids), sources=len(reference.source_ids),
        positives=sum(sum(v > 0 for v in labels[q].values()) for q in chosen),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        hard_label_sha256=base.digest(root / f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=base.digest(root / "qwen-embedding-torch-v1/protocol.json"),
        code_sha256=base.digest(Path(__file__)), embedding_revision=provenance["revision"],
        reranker_revision=verifier.provenance["revision"], arms=summarize(rows, [labels[q] for q in chosen]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("SIGIR", "KDD", "SciRepEval"))
    for field in ("embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
