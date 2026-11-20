"""Separate technical evidence recovery from person attribution on owned-source gaps."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from experiments.retrieval import enterprise_gap_bound_recovery as previous
from experiments.retrieval.enterprise_declaration_reach import CapturedIndex
from experiments.retrieval.enterprise_project_role_rescue import terms
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("role_only", "lexical_owned", "dense_owned")
CAP = 2
WINDOW = 520
STEP = 260
BOUNDARY = re.compile(r"[\r\n;\uff1b\u3002.!?]+")


def windows(text, lower):
    if len(text) <= lower:
        return []
    last = max(lower, len(text) - WINDOW)
    offsets = {lower, last, *range(lower, last, STEP)}
    offsets.update(m.end() for m in BOUNDARY.finditer(text) if lower <= m.end() <= last)
    return [(a, min(len(text), a + WINDOW)) for a in sorted(offsets)]


def overlap(criterion, text):
    words = terms(text)
    return len(criterion & words) / max(1, len(words)) ** .5


class OwnedSearch:
    def __init__(self, sources, vectors, role_windows, encoder):
        self.sources, self.vectors, self.encoder = sources, vectors, encoder
        self.source_number = {source: i for i, source in enumerate(sources)}
        self.owned = defaultdict(list)
        self.role = defaultdict(list)
        self.named = defaultdict(list)
        for key, row in sources.items():
            self.owned[row["expert_id"]].append(key)
        for row in role_windows:
            cite = row["view"]
            span = cite["id"], cite["start"], cite["end"]
            self.role[row["expert"]].append(span)
            if set(cite["roles"]) & {"profile_named_action", "profile_listed_inventor"}:
                self.named[cite["id"]].append((cite["start"], cite["end"]))

    def excerpt(self, citation):
        key, a, b = citation
        return self.sources[key]["text"][a:b]

    def has_named_cue(self, citation):
        key, a, b = citation
        return any(a <= left < right <= b for left, right in self.named[key])

    def best(self, query_terms, source, lower):
        text = self.sources[source]["text"]
        choices = windows(text, lower)
        if not choices:
            return None
        a, b = max(choices, key=lambda pair: (overlap(query_terms, text[pair[0]:pair[1]]), -pair[0]))
        return source, a, b

    def candidates(self, owner, search_terms, vector, originals, scoped, arm, ceiling):
        if arm == "role_only":
            choices = [c for c in self.role[owner] if ((not scoped and c[0] not in originals)
                       or (scoped and c[0] in scoped and c[1] >= ceiling))]
        else:
            sources = ([source for source in self.owned[owner] if source not in originals]
                       if not scoped else [source for source in scoped if source in self.owned[owner]])
            choices = [c for source in sources if (c := self.best(search_terms, source, ceiling if scoped else 0))]
        lexical = {c: overlap(search_terms, self.excerpt(c)) for c in choices}
        if arm == "dense_owned" and choices:
            indexes = [self.source_number[c[0]] for c in choices]
            affinity = dict(zip(choices, self.vectors[indexes] @ vector, strict=True))
            return sorted(choices, key=lambda c: (-affinity[c], -lexical[c], c))[:CAP]
        return sorted((c for c in choices if lexical[c] > 0.), key=lambda c: (-lexical[c], c))[:CAP]


def analyze(arm, criteria, partial, original_sources, shared_scope, owned, verifier, search):
    opened = perf_counter()
    target_ids = sorted({i for _, spans in partial for i, citation in enumerate(spans) if not citation})
    if arm == "dense_owned" and target_ids:
        # Encode all distinct missing requirements in this request in one live call.
        prompt = f"Instruct: {search.base.EMBED_TASK}\nQuery: "
        embedded = dict(zip(target_ids, owned.encoder.encode([prompt + criteria[i] for i in target_ids]), strict=True))
    else:
        embedded = {}
    cues = {i: terms(criteria[i]) for i in target_ids}
    checks, span_map, missing = [], [], {}
    for owner, prior in partial:
        scoped = {p[0] for p in prior if p} if shared_scope else ()
        if len(scoped) > 1:
            continue
        missing[owner] = [i for i, citation in enumerate(prior) if not citation]
        for i in missing[owner]:
            options = owned.candidates(owner, cues[i], embedded.get(i), original_sources[owner],
                                       scoped, arm, search.base.SOURCE_CHARS)
            for cite in options:
                checks.append((criteria[i], owned.excerpt(cite)))
                span_map.append((owner, i, cite))
    passed = {}
    scored = []
    clipped = 0
    for offset in range(0, len(checks), search.base.VERIFY_BATCH):
        batch = checks[offset:offset+search.base.VERIFY_BATCH]
        margins, truncated = verifier.score_pairs(batch, task=search.base.TASK)
        if truncated or not np.isfinite(margins).all():
            clipped += int(truncated) or len(batch)
            continue
        for record, margin in zip(span_map[offset:offset+len(batch)], margins, strict=True):
            owner, i, cite = record
            scored.append((float(margin), owner, i, cite))
            if margin > 0:
                # Prefer named evidence when both passages get positive technical scores.
                rank = owned.has_named_cue(cite), margin
                if (owner, i) not in passed or rank > passed[owner, i][0]:
                    passed[owner, i] = rank, cite
    completed = []
    for owner, prior in partial:
        targets = missing.get(owner)
        if targets and all((owner, i) in passed for i in targets):
            additions = [passed[owner, i][1] for i in targets]
            citations = list(dict.fromkeys([*[p for p in prior if p], *additions]))
            completed.append(dict(expert=owner, citations=citations, new_citations=additions,
                                  new_evidence_all_named=all(owned.has_named_cue(c) for c in additions),
                                  whole_bundle_all_named=all(owned.has_named_cue(c) for c in citations)))
    review_candidates = []
    if scored:
        _, owner, _, citation = max(scored)
        original = [point for person, cites in partial if person == owner for point in cites if point]
        review_candidates.append(dict(expert=owner, citations=list(dict.fromkeys([*original, citation])),
                                      new_citations=[citation]))
    return dict(checks=len(checks), positive_criteria=len(passed), clipped=clipped,
                near_zero_nonpositive=sum(-1. <= entry[0] <= 0. for entry in scored),
                review_candidates=review_candidates,
                positive_criteria_with_named_cue=sum(rank[0] for rank, _ in passed.values()),
                completed=completed, ms=(perf_counter()-opened)*1000)


def run(queries, corpus, roles, vectors, vector_meta, models, output, trace):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import enterprise_shared_qualification as search
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    search.role.require_cuda(output, trace)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Unexpected development inventory")
    sources = {row["id"]: row for row in rows(corpus)}
    meta = json.loads(vector_meta.read_bytes())
    model_path = models/"qwen3-embedding-0.6b"
    prov = verify_checkpoint(model_path)
    if (SOURCE_INVENTORY, prov["revision"], digest(vectors), len(sources)) != tuple(
            meta[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    matrix = np.load(vectors, allow_pickle=False)
    index = CapturedIndex(GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        matrix, list(sources), [[row["expert_id"]] for row in sources.values()])))
    parser = ComparisonParser(models/"qwen3-4b-instruct-2507")
    reader = search.EncodedReader(Reranker(models/"qwen3-reranker-0.6b"))
    encoder = QwenTorchEmbedding(model_path)
    role_index = search.role.load_role_index(roles, SOURCE_INVENTORY, sources)
    owned = OwnedSearch(sources, matrix, role_index[0], encoder)
    prefix = parser, encoder, index, sources, reader
    previous.primary("Find an expert in optical sensor calibration.", *prefix)
    reports = []
    for n, item in enumerate(rows(queries)):
        text = item["text"]
        criteria, _, decisions, _ = previous.primary(text, *prefix)
        if any(approved for _, approved, _ in decisions) or any(len(c)>search.base.CRITERION_CHARS for c in criteria):
            continue
        partial = [(owner, citations) for owner, _, citations in decisions if any(citations) and not all(citations)]
        if not partial:
            continue
        original_sources = {hit["id"]: {src["id"] for src in hit["evidence"][:search.base.SOURCE_DEPTH]}
                            for hit in index.hits}
        scoped = bool(previous.COMMON_OBJECT.search(text))
        offset = len(reports) % len(ARMS)
        arms = {arm: analyze(arm, criteria, partial, original_sources, scoped, owned, reader, search)
                for arm in ARMS[offset:] + ARMS[:offset]}
        limit = CAP * sum(not c for _, cites in partial for c in cites)
        if any(result["checks"] > limit for result in arms.values()):
            raise ValueError("A route exceeded the fixed check cap")
        reports.append(dict(id=item["id"], arms=arms, scoped=scoped, partial=len(partial), check_limit=limit))
    stats = {}
    for arm in ARMS:
        measures = [row["arms"][arm] for row in reports]
        tally = Counter()
        for arm_result in measures:
            tally.update({field: arm_result[field]
                          for field in ("checks","positive_criteria","positive_criteria_with_named_cue","clipped","near_zero_nonpositive")})
            completed = arm_result["completed"]
            tally.update(requests_with_complete_technical_bundle=bool(completed), complete_technical_bundles=len(completed),
                         new_evidence_all_named=sum(r["new_evidence_all_named"] for r in completed),
                         whole_bundle_all_named=sum(r["whole_bundle_all_named"] for r in completed),
                         bundles_with_new_paper=sum(any(c[0].startswith("paper:") for c in r["new_citations"])
                                                    for r in completed))
        stats[arm] = dict(tally) | dict(recovery_p95_ms=float(np.quantile([r["ms"] for r in measures],.95)))
    write(trace, dict(source_sha256=SOURCE_INVENTORY, query_sha256=DEVELOPMENT, requests=reports))
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       code_sha256=digest(Path(__file__)), trace_sha256=digest(trace),
                       eligible_requests=len(reports), candidate_slots=sum(r["partial"] for r in reports),
                       maximum_pairs=sum(r["check_limit"] for r in reports),
                       common_object_requests=sum(r["scoped"] for r in reports), arms=stats))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "vector_meta", "models", "output", "trace"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
