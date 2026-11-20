"""Use already supported criteria when reading a new role-evidence window."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("global_full", "reuse", "gap_reserve")
COMMON_OBJECT = re.compile(r"\bsame\b|single (?:project|system|site|device|study)|\u540c\u4e00|\u540c\u4e00\u4e2a", re.I)


def primary(text, parser, encoder, index, sources, reader):
    from experiments.retrieval import enterprise_shared_qualification as search
    from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans

    start = perf_counter()
    clauses = spans(text)
    parsed = parser.evaluate(text, clauses, "broadcast_prefix", cutoff=search.base.CUTOFF)
    qualifiers = tuple((c["start"], c["end"]) for c, label in zip(clauses, parsed["labels"], strict=True)
                       if ROLES[int(label)] == "prior_qualification")
    criteria = (text.strip().splitlines()[0][:search.base.CRITERION_CHARS], *[text[a:b] for a, b in qualifiers])
    vector = encoder.encode([f"Instruct: {search.base.EMBED_TASK}\nQuery: {text}"])[0]
    hits = index.search_vector(vector, k=search.base.K)["hits"]
    reader.change("guarded_rescue")
    decisions, _ = search.execute("automatic", hits, sources, criteria, reader)
    return criteria, qualifiers, decisions, (perf_counter() - start) * 1000


def citations_for(item, prior, scoped):
    witness = prior.get(item["expert"], ())
    if scoped and any(citation and citation[0] != item["view"]["id"] for citation in witness):
        return ()
    return witness


def reserve(global_options, windows, postings, prior, criterion_terms, scoped):
    if len(global_options) < 2:
        return global_options
    first = global_options[0]
    other = None
    for owner, witness in prior.items():
        if owner == first["expert"] or not any(witness) or all(witness):
            continue
        holes = [i for i, citation in enumerate(witness) if not citation]
        terms = set().union(*(criterion_terms[i] for i in holes))
        options = set().union(*(postings.get(term, ()) for term in terms))
        for i in options:
            row = windows[i]
            if row["expert"] != owner or not citations_for(row, prior, scoped):
                continue
            local = row["terms"]
            cover = [len(criterion_terms[j] & local) / max(1, len(criterion_terms[j])) for j in holes]
            if not all(cover):
                continue
            score = (.65 * min(cover) + .35 * mean(cover), -len(holes), -i)
            if other is None or score > other[0]:
                other = score, row
    return [first, {k: other[1][k] for k in ("expert", "view")}] if other else global_options


def recover(arm, criteria, decisions, role_index, verifier, scoped):
    from experiments.retrieval import enterprise_shared_qualification as search

    started = perf_counter()
    prior = {owner: tuple(citations) for owner, _, citations in decisions}
    existing = [dict(expert=owner, citations=list(dict.fromkeys(citations)))
                for owner, approved, citations in decisions if approved]
    limit = search.base.CRITERION_CHARS
    if existing or any(len(criterion) > limit for criterion in criteria):
        return dict(issued=existing, extra=[], checks=0, windows=0, reused=0, rescue_ms=0.,
                    replacement=False, clipped=0)
    windows, postings = role_index
    terms = list(map(search.role.role.terms, criteria))
    hits = {i for term in terms[0] for i in postings.get(term, ())}
    ranked = {owner: rank for rank, owner in enumerate(prior)}
    global_options = search.role.expanded.select((windows[i] for i in sorted(hits)), criteria,
                                                 ranked)[:search.role.ROLE_READS]
    selected = reserve(global_options, windows, postings, prior, terms, scoped) if arm == "gap_reserve" else global_options
    verifier.counts.clear()
    inputs, actions = [], []
    reused = 0
    for item in selected:
        prior_evidence = citations_for(item, prior, scoped) if arm != "global_full" else ()
        full = list(prior_evidence or (None,) * len(criteria))
        missing = [i for i, source in enumerate(full) if source is None]
        reused += len(full) - len(missing)
        inputs.extend((criteria[i], item["view"]["text"]) for i in missing)
        actions.append((item, full, missing))
    values, clipped = verifier.score_pairs(inputs, task=search.base.TASK) if inputs else ([], 0)
    if len(values) != len(inputs) or not np.isfinite(values).all():
        raise ValueError("Invalid fallback output")
    accepted = {}
    offset = 0
    if not clipped:
        for item, full, missing in actions:
            end = offset + len(missing)
            if all(score > 0. for score in values[offset:end]):
                view = item["view"]
                for i in missing:
                    full[i] = view["id"], view["start"], view["end"]
                accepted.setdefault(item["expert"], dict(expert=item["expert"],
                                                         citations=list(dict.fromkeys(full))))
            offset = end
    extra = list(accepted.values())[:search.base.K]
    return dict(issued=extra, extra=extra, checks=len(inputs), windows=len(selected), reused=reused,
                rescue_ms=(perf_counter() - started) * 1000,
                replacement=selected != global_options, clipped=clipped)



def exhaustive(criteria, decisions, by_owner, verifier, scoped):
    from experiments.retrieval import enterprise_shared_qualification as search

    result = Counter()
    if any(approved for _, approved, _ in decisions) or any(len(c) > search.base.CRITERION_CHARS
                                                          for c in criteria):
        return result
    partial = {owner: tuple(witness) for owner, _, witness in decisions if any(witness) and not all(witness)}
    result["eligible_requests"] += bool(partial)
    result["initial_partial_owners"] += len(partial)
    completed = set()
    with_window = set()
    observations = []
    for owner, witness in partial.items():
        for item in by_owner.get(owner, ()):
            if not citations_for(item, partial, scoped):
                result["skipped_scope_windows"] += 1
                continue
            missing = [i for i, source in enumerate(witness) if source is None]
            observations.append((owner, [(criteria[i], item["view"]["text"]) for i in missing]))
            with_window.add(owner)
    result["owners_with_role_window"] += len(with_window)
    result["requests_with_role_window"] += bool(with_window)
    for start in range(0, len(observations), 8):
        entries = observations[start:start+8]
        pairs = [pair for _, items in entries for pair in items]
        scores, clipped = verifier.score_pairs(pairs, task=search.base.TASK)
        if clipped or not np.isfinite(scores).all():
            result["unscored_windows"] += len(entries)
            continue
        offset = 0
        for owner, points in entries:
            end = offset + len(points)
            accepted = all(x > 0. for x in scores[offset:end])
            if accepted:
                completed.add(owner)
            result.update(windows=1, criterion_checks=len(points), windows_filling_all_gaps=int(accepted))
            offset = end
    result["provisionally_completed_owners"] += len(completed)
    result["requests_with_provisionally_completed_owner"] += bool(completed)
    return result


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
    model_path = models / "qwen3-embedding-0.6b"
    prov = verify_checkpoint(model_path)
    if (meta["corpus_sha256"], meta["checkpoint"], meta["vectors_sha256"], meta["rows"]) != (
            SOURCE_INVENTORY, prov["revision"], digest(vectors), len(sources)):
        raise ValueError("Vector provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[row["expert_id"]] for row in sources.values()]))
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    reader = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    encoder = QwenTorchEmbedding(model_path)
    role_index = search.role.load_role_index(roles, SOURCE_INVENTORY, sources)
    prefix = (parser, encoder, index, sources, reader)
    by_owner = defaultdict(list)
    for item in role_index[0]:
        by_owner[item["expert"]].append(item)
    audit = Counter()
    warm = "Find optical inspection expertise with prior work in sensor calibration."
    criteria, _, decisions, _ = primary(warm, *prefix)
    for arm in ARMS:
        recover(arm, criteria, decisions, role_index, reader, False)
    reports = []
    for i, item in enumerate(rows(queries)):
        criteria, qualifications, decisions, primary_ms = primary(item["text"], *prefix)
        scoped = bool(COMMON_OBJECT.search(item["text"]))
        order = i % len(ARMS)
        results = {arm: recover(arm, criteria, decisions, role_index, reader, scoped)
                   for arm in ARMS[order:] + ARMS[:order]}
        if len({result["windows"] for result in results.values()}) != 1:
            raise ValueError("Window count changed")
        control = results["global_full"]
        if any(arm["checks"] > control["checks"] for arm in results.values()):
            raise ValueError("Fallback exceeded the reference check budget")
        audit.update(exhaustive(criteria, decisions, by_owner, reader, scoped))
        reports.append(dict(id=item["id"], arms=results, primary_ms=primary_ms, qualifications=qualifications,
                            scoped=scoped, has_partial=any(any(c) and not all(c) for _, _, c in decisions)))
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1} development requests", flush=True)
    results = {}
    for arm in ARMS:
        counts = Counter()
        times, opened = [], []
        for row in reports:
            ref, val = row["arms"]["global_full"], row["arms"][arm]
            counts.update({key: val[key] for key in ("checks", "reused", "windows", "clipped")})
            counts.update(provisional_issued_queries=bool(val["issued"]), provisional_issued_slots=len(val["issued"]),
                          fallback_recommendations=len(val["extra"]), changed_outputs=val["issued"] != ref["issued"],
                          new_answer=bool(val["issued"] and not ref["issued"]),
                          lost_answer=bool(ref["issued"] and not val["issued"]),
                          query_with_reused_check=bool(val["reused"]), replacement=int(val["replacement"]),
                          multisource_promotions=sum(len({p[0] for p in hit["citations"]}) > 1 for hit in val["extra"]))
            times.append(row["primary_ms"] + val["rescue_ms"])
            if val["windows"]:
                opened.append(val["rescue_ms"])
        results[arm] = dict(counts) | dict(shared_primary_plus_rescue_p95_ms=float(np.quantile(times, .95)),
                                          fallback_p95_ms=float(np.quantile(opened, .95)) if opened else None)
    write(trace, dict(source_sha256=SOURCE_INVENTORY, query_sha256=DEVELOPMENT, reports=reports))
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       script_sha256=digest(Path(__file__)), query_count=len(reports), source_count=len(sources),
                       trace_sha256=digest(trace), scopes=sum(row["scoped"] for row in reports),
                       partial_criteria_queries=sum(row["has_partial"] for row in reports),
                       arms=results, exhaustive_owner_role_windows=dict(audit)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "vector_meta", "models", "output", "trace"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
