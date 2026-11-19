"""Reuse identical verifier inputs and skip disqualified candidate rows."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.retrieval import enterprise_first_query as base
from experiments.retrieval import enterprise_role_first_query as role
from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("guarded", "shared", "qualification_first", "automatic")
LARGE = 32
RANK_GUARD = .125


@dataclass(frozen=True)
class Pair:
    expert: str
    criterion: int
    text: str
    evidence: str
    citation: tuple

    @property
    def model_key(self):
        return self.text, self.evidence


def plan(hits, sources, criteria):
    pairs = []
    for hit in hits:
        expert = hit["id"]
        for item in hit["evidence"][:base.SOURCE_DEPTH]:
            doc = sources[item["id"]]
            if doc["expert_id"] != expert:
                raise ValueError("Citation is not linked to the candidate")
            excerpt = doc["text"][:base.SOURCE_CHARS]
            if excerpt:
                citation = (item["id"], 0, len(excerpt))
                pairs.extend(Pair(expert, i, criterion, excerpt, citation)
                             for i, criterion in enumerate(criteria) if len(criterion) <= base.CRITERION_CHARS)
    return pairs


def exact(hits, sources, criteria, reader):
    return base.verify_candidates(hits, sources, criteria, reader)[0]


class EncodedReader(packed.Reader):
    def score_encoded(self, prepared):
        with self.torch.inference_mode():
            lengths = [len(row["input_ids"]) for row in prepared]
            actual = sum(lengths)
            padded = max(lengths) * len(lengths)
            self.counts.update(logical_pairs=len(prepared), logical_tokens=actual)
            if len(lengths) > 1 and actual < packed.TOKEN_RATIO * padded:
                margins, tokens = self.forward(self.backbone, self.weight, prepared)
                self.counts.update(packed_batches=1, executed_tokens=tokens)
                return margins
            return self.padded(prepared, padded)


def scores(keys, encoding, reader):
    margins = {}
    for start in range(0, len(keys), base.VERIFY_BATCH):
        batch = keys[start:start + base.VERIFY_BATCH]
        values = reader.score_encoded([encoding[key] for key in batch])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite support margins")
        margins.update(zip(batch, values, strict=True))
    return margins


def choose(pairs, margins, hits, needed):
    candidate_witnesses = defaultdict(dict)
    minimum_gap = float("inf")
    for pair in pairs:
        margin = margins.get(pair.model_key)
        if margin is None or margin <= 0:
            continue
        witnesses = candidate_witnesses[pair.expert]
        previous = witnesses.get(pair.criterion)
        if previous is not None and pair.citation != previous[1]:
            minimum_gap = min(minimum_gap, abs(margin - previous[0]))
        if previous is None or margin > previous[0]:
            witnesses[pair.criterion] = margin, pair.citation
    result = []
    for hit in hits:
        expert = hit["id"]
        witnesses = candidate_witnesses[expert]
        citations = tuple(witnesses.get(i, (None, None))[1] for i in range(needed))
        result.append((expert, all(citations), citations))
    result.sort(key=lambda row: not row[1])
    return result, minimum_gap


def execute(arm, hits, sources, criteria, reader):
    proposed = plan(hits, sources, criteria)
    keys = list(dict.fromkeys(pair.model_key for pair in proposed))
    duplicates = len(proposed) - len(keys)
    stage = (len(criteria) > 1 and arm in ("qualification_first", "automatic")
             and (arm == "qualification_first" or len(proposed) >= LARGE))
    metadata = dict(original_pairs=len(proposed), exact_duplicates=duplicates, removed_due_to_objective=0,
                    staged=int(stage), fallback=0, fallback_reason="")
    if arm == "guarded" or (not stage and not duplicates):
        return exact(hits, sources, criteria, reader), metadata

    prepared, truncated = reader.encode(keys, base.TASK)
    if truncated:
        metadata.update(fallback=1, fallback_reason="context")
        return exact(hits, sources, criteria, reader), metadata
    encoding = dict(zip(keys, prepared, strict=True))
    if stage:
        required = list(dict.fromkeys(pair.model_key for pair in proposed if pair.criterion == 0))
        margins = scores(required, encoding, reader)
        viable = {p.expert for p in proposed if p.criterion == 0 and margins[p.model_key] > 0}
        final_keys = list(dict.fromkeys(p.model_key for p in proposed if p.expert in viable and p.model_key not in margins))
        removed = len(keys) - len(required) - len(final_keys)
        metadata["removed_due_to_objective"] = removed
        margins.update(scores(final_keys, encoding, reader))
    else:
        margins = scores(keys, encoding, reader)

    result, rank_gap = choose(proposed, margins, hits, len(criteria))
    near_threshold = any(abs(value) <= packed.SIGN_MARGIN for value in margins.values())
    if near_threshold or rank_gap <= RANK_GUARD:
        metadata.update(fallback=1, fallback_reason="sign_or_witness_gap")
        return exact(hits, sources, criteria, reader), metadata
    return result, metadata


def one(text, arm, parser, embedder, index, sources, role_index, reader):
    from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans

    started = perf_counter()
    clauses = spans(text)
    parsed = parser.evaluate(text, clauses, "broadcast_prefix", cutoff=base.CUTOFF)
    qualifications = tuple((c["start"], c["end"]) for c, label in zip(clauses, parsed["labels"], strict=True)
                           if ROLES[int(label)] == "prior_qualification")
    criteria = (text.strip().splitlines()[0][:base.CRITERION_CHARS],) + tuple(text[a:b] for a,b in qualifications)
    vector = embedder.encode([f"Instruct: {base.EMBED_TASK}\nQuery: {text}"])[0]
    hits = index.search_vector(vector, k=base.K)["hits"]
    verifier_started = perf_counter()
    reader.change("guarded_rescue")
    decisions, metadata = execute(arm, hits, sources, criteria, reader)
    verifier_ms = (perf_counter() - verifier_started) * 1000
    primary = dict(reader.counts)
    initial = [dict(expert=eid, citations=[list(span) for span in dict.fromkeys(citations)])
               for eid, accepted, citations in decisions if accepted]
    ranked = [eid for eid, _, _ in decisions]
    reader.counts.clear()
    extra, _, invoked, clipped = role.recover(
        criteria, ranked, {item["expert"] for item in initial}, role_index,
        reader, "global_on_abstain", base.TASK, base.K - len(initial)
    ) if all(len(c) <= base.CRITERION_CHARS for c in criteria) else ([], 0, False, 0)
    return dict(
        initial=initial, extra=extra, decisions=decisions, ranked=ranked,
        retrieved=[hit["id"] for hit in hits], qualifications=qualifications, invoked=invoked,
        metadata=metadata, primary=primary, fallback=dict(reader.counts), fallback_clipped=clipped,
        verifier_ms=verifier_ms, elapsed_ms=(perf_counter() - started) * 1000,
    )


def summarize(observations):
    reference = [r["guarded"] for r in observations]
    boot = np.random.default_rng(20261120).integers(len(reference), size=(6000, len(reference)))
    base_elapsed = np.asarray([r["elapsed_ms"] for r in reference])
    base_q95 = np.quantile(base_elapsed[boot], .95, axis=1)
    summary = {}
    for arm in ARMS:
        values = [o[arm] for o in observations]
        elapsed = np.asarray([r["elapsed_ms"] for r in values])
        report = summary[arm] = dict(
            p50_p95_ms=np.quantile(elapsed, [.50, .95]).tolist(),
            paired_p95_delta_vs_guarded=np.quantile(np.quantile(elapsed[boot], .95, axis=1)-base_q95,[.025,.975]).tolist(),
            primary_verifier_p95_ms=float(np.quantile([r["verifier_ms"] for r in values], .95)),
            changes={key: sum(a[key] != b[key] for a,b in zip(values, reference, strict=True))
                     for key in ("initial", "extra", "ranked", "decisions", "retrieved", "qualifications", "invoked")},
            issued=sum(bool(r["initial"] or r["extra"]) for r in values),
            fallback_guard_reasons=dict(Counter(r["metadata"]["fallback_reason"] for r in values if r["metadata"]["fallback"])),
        )
        for stage in ("primary", "fallback"):
            report[stage] = dict(sum((Counter(r[stage]) for r in values), Counter()))
        for field in ("original_pairs", "exact_duplicates", "removed_due_to_objective", "staged", "fallback"):
            report[field if field!="fallback" else "canonical_rechecks"] = sum(r["metadata"][field] for r in values)
    return summary


def run(queries, corpus, roles, vectors, meta, models, output):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Expected the pinned development-only inputs")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {source["id"]: source for source in rows(corpus)}
    saved = json.loads(meta.read_bytes())
    encoder_path = models / "qwen3-embedding-0.6b"
    info = verify_checkpoint(encoder_path)
    if (SOURCE_INVENTORY, info["revision"], digest(vectors), len(sources)) != tuple(
            saved[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector cache provenance mismatch")
    embedding = QwenTorchEmbedding(encoder_path)
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    role_index = role.load_role_index(roles, SOURCE_INVENTORY, sources)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    canonical = Reranker(models / "qwen3-reranker-0.6b")
    verifier = EncodedReader(canonical)
    fixture = parser, embedding, index, sources, role_index, verifier
    for arm in ARMS:
        one("Find an expert in optical inspection with prior polymer testing experience.", arm, *fixture)
    observations = []
    for i, request in enumerate(rows(queries)):
        shift = i % len(ARMS)
        observations.append({arm: one(request["text"], arm, *fixture) for arm in ARMS[shift:] + ARMS[:shift]})
        if (i + 1) % 20 == 0:
            print(f"Measured {i+1} private development requests", flush=True)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(observations), code_sha256=digest(Path(__file__)), queries_sha256=DEVELOPMENT,
        source_sha256=SOURCE_INVENTORY, role_windows_sha256=digest(roles),
        models=dict(encoder=info["revision"], verifier=canonical.provenance["revision"],
                    parser=parser.readers["bf16"].provenance["revision"]), metrics=summarize(observations)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
