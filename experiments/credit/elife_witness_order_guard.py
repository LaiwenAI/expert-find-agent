"""Measure when compacted GPU scores preserve ordered eLife expert citations."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.requirement_packets import form_packet
from expert_search.witness_order import stable_witnesses
from experiments.credit import elife_proof_before_rerank as search
from experiments.credit import elife_role_normalization_search as normalization
from experiments.credit import packed_topic_guard as prior
from experiments.credit import stable_topic_compaction as compaction
from experiments.credit import study_role_normalization as roles
from experiments.credit import study_role_normalization_batch as batch
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


ARMS = ("full", "compact", "fixed_width", "cutoff_guard", "prior_guard", "witness_guard")
GUARDS = {"cutoff_guard", "prior_guard", "witness_guard"}
TOP = normalization.SEATS
CUTOFF = holdout.CUTOFF


def witnesses(scores, owners, *, admitted=None):
    emitted = {}
    for position in sorted(scores, key=lambda p: (-scores[p], p)):
        if not (scores[position] > CUTOFF if admitted is None else position in admitted):
            continue
        for owner in owners[position]:
            emitted.setdefault(owner, position)
            if len(emitted) == TOP:
                return tuple(emitted.items())
    return tuple(emitted.items())


def owner_lists(catalog, ranked, fields):
    owners = {}
    for position, source in enumerate(ranked):
        article = catalog.articles[source]
        owners[position] = tuple(
            row["person"] or f'{article["doi"]}#{row["ordinal"]}'
            for row in article["authors"] if fields.issubset(row["roles"])
        )
    return owners


def prepare(catalog, judge, reader, query):
    topic, _, text, _ = query
    start = perf_counter()
    packet = form_packet(text, None)
    clauses = [span.quote(text) for span in packet.unresolved]
    if not clauses or not all(clause.startswith(search.ROLE_PREFIX) for clause in clauses):
        return None
    tasks = [clause.removeprefix(search.ROLE_PREFIX).removesuffix(".") for clause in clauses]
    _, accepted, _ = batch.normalize(judge, tasks, "one_batch")
    if any(value not in normalization.NATIVE for value in accepted) or len(set(accepted)) != len(tasks):
        return None
    fields = frozenset(normalization.NATIVE[value] for value in accepted)
    ranked = catalog.rank(topic)[:8]
    rows = reader.encode(f"Research about {topic}", [catalog.texts[source] for source in ranked])
    return ranked, rows, owner_lists(catalog, ranked, fields), (perf_counter() - start) * 1000


def options(ranked, visible, owners, limit):
    return [p for p, source in enumerate(ranked[:limit]) if source not in visible or owners[p]]


def score(reader, rows, positions, width=None):
    values, tokens = reader.score(rows, positions, width=width)
    return dict(zip(positions, values, strict=True)), tokens


def guard_requires_recheck(arm, scores, owners, radius):
    if arm == "cutoff_guard":
        return any(owners[p] and abs(value - CUTOFF) <= radius for p, value in scores.items())
    if arm == "prior_guard":
        return prior.output_risk(scores, owners, radius)
    return not stable_witnesses(scores, owners, radius=radius, cutoff=CUTOFF, top=TOP)


def run_arm(reader, rows, owners, active, radius, arm, common_ms):
    start = perf_counter()
    positions = list(range(len(rows)))
    chosen = positions if arm == "full" else active
    width = max(len(row["input_ids"]) for row in rows) if arm == "fixed_width" else None
    scores, tokens = score(reader, rows, chosen, width)
    rechecked = False
    used = scores
    if arm in GUARDS and chosen != positions and guard_requires_recheck(arm, scores, owners, radius):
        used, extra = score(reader, rows, positions)
        tokens += extra
        rechecked = True
    result = witnesses(used, owners)
    return dict(result=result, approx=scores, tokens=tokens, rechecked=rechecked,
                rows=len(chosen) + rechecked * len(positions), ms=common_ms + (perf_counter() - start) * 1000)


def calibrate(reader, prepared, settings):
    largest = 0.
    compared = 0
    for ranked, rows, owners, _ in prepared:
        for limit in search.LIMITS:
            reference, _ = score(reader, rows[:limit], list(range(limit)))
            for visible in settings.values():
                active = options(ranked, visible, owners, limit)
                smaller, _ = score(reader, rows[:limit], active)
                errors = [abs(value - reference[p]) for p, value in smaller.items()]
                largest = max([largest, *errors])
                compared += len(errors)
    if not compared:
        raise ValueError("Empty calibration")
    return largest, compared


def audit(result, ref, owners, radius):
    approx = result["approx"]
    original = {p: ref["approx"][p] for p in approx}
    before = {p for p, value in original.items() if value > CUTOFF}
    after = {p for p, value in approx.items() if value > CUTOFF}
    errors = [abs(value - original[p]) for p, value in approx.items()]
    if witnesses(original, owners) != ref["result"]:
        raise AssertionError("Pruned a source that could emit a witness")
    order_only = witnesses(approx, owners, admitted=before) != ref["result"]
    cutoff_only = witnesses(original, owners, admitted=after) != ref["result"]
    differs_raw = witnesses(approx, owners) != ref["result"]
    return dict(
        changed_output=result["result"] != ref["result"], raw_changed=differs_raw,
        threshold_flips=len(before.symmetric_difference(after)), output_cutoff_only=cutoff_only,
        output_order_only=order_only, interaction_only=differs_raw and not cutoff_only and not order_only,
        envelope_exceedances=sum(error > radius for error in errors),
        max_abs_error=max(errors, default=0.),
    )


def evaluate(catalog, judge, reader, offset):
    complete = {i for i, article in enumerate(catalog.articles) if article["declaration_complete"]}
    settings = {"partial": complete & catalog.visible[50], "full": complete}
    query_panel = [q for q in search.panel(catalog.topics, search.postings(catalog.articles), complete) if not q[-1]]
    topic_keys = sorted({q[0] for q in query_panel}, key=lambda key: pilot.stable("witness-interval-v1", key))
    development = set(topic_keys[::2])
    sample = next(query for query in query_panel if query[0] in development)
    warm = prepare(catalog, judge, reader, sample)
    if warm:
        score(reader, warm[1], list(range(8)))
    calibration = [row for query in query_panel if query[0] in development
                   if (row := prepare(catalog, judge, reader, query)) is not None]
    radius, compared = calibrate(reader, calibration, settings)
    outcomes = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for j, query in enumerate(q for q in query_panel if q[0] not in development):
        row = prepare(catalog, judge, reader, query)
        if row is None:
            skipped += 1
            continue
        ranked, rows, owners, common_ms = row
        for mask, visible in settings.items():
            for limit in search.LIMITS:
                active = options(ranked, visible, owners, limit)
                shift = (j + offset) % len(ARMS)
                trials = {arm: run_arm(reader, rows[:limit], owners, active, radius, arm, common_ms)
                          for arm in ARMS[shift:] + ARMS[:shift]}
                for arm, result in trials.items():
                    counts = audit(result, trials["full"], owners, radius)
                    if (arm == "witness_guard" and not counts["envelope_exceedances"]
                            and counts["changed_output"]):
                        raise AssertionError("Witness order changed within certified score intervals")
                    outcomes[f"{mask}/{limit}"][arm].append((query[0], counts, result))

    reports = {}
    for setting, arms in outcomes.items():
        reports[setting] = {}
        ref_ms = [r["ms"] for _, _, r in arms["full"]]
        for arm, rows in arms.items():
            counts = Counter()
            for _, audit_row, result in rows:
                counts.update({key: value for key, value in audit_row.items() if key != "max_abs_error"})
                counts.update(rechecked=result["rechecked"], token_slots=result["tokens"], reader_rows=result["rows"])
            elapsed = [r["ms"] for _, _, r in rows]
            reports[setting][arm] = dict(
                queries=len(rows), **counts,
                max_abs_error=max(a["max_abs_error"] for _, a, _ in rows),
                p95_ms=float(np.quantile(elapsed, .95)),
                paired_p95_delta_ci95_ms=holdout.compare_latency(elapsed, ref_ms, [topic for topic, _, _ in rows]),
            )
    return dict(calibration_topics=len(development), calibration_queries=len(calibration),
                calibration_scores=compared, tolerance=radius, test_topics=len(topic_keys) - len(development),
                test_abstained_before_scoring=skipped, report=reports)


def main():
    import torch
    from experiments.followir.core17_verifier import Reranker

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pinned = json.loads(args.manifest.read_bytes())
    reference = json.loads(args.reference.read_bytes())["run"]
    if digest(args.projection) != pinned["projection_sha256"]:
        raise ValueError("Unexpected eLife projection")
    embedding = Path("artifacts/models/qwen3-embedding-0.6b")
    provenance = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    ranker = Reranker(Path("artifacts/models/qwen3-reranker-0.6b"))
    if (provenance["revision"], ranker.provenance["revision"]) != (reference["embedding"], reference["reranker"]):
        raise ValueError("Unexpected topical checkpoint")
    judge = Judge()
    roles.extended_ids(judge)
    articles = [dict(row, topics=row["keywords"]) for row in json.loads(args.projection.read_bytes())["sources"]]
    catalog = pilot.Catalog(articles, encoder, ranker)
    result = evaluate(catalog, judge, compaction.BatchReader(ranker), args.order_offset)
    write_json(args.output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
                                 certificate_sha256=digest(Path(stable_witnesses.__code__.co_filename)),
                                 projection_sha256=digest(args.projection), decision_model_sha256=judge.sha, **result))


if __name__ == "__main__":
    main()
