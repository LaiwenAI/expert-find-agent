"""Diagnose expert and evidence headroom on fixed multi-condition requests."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.retrieval import enterprise_hybrid_rescue as hybrid
from experiments.retrieval import enterprise_role_first_query as roles
from experiments.retrieval.enterprise_joint_first_round import prepare_owned, prefixes
from experiments.retrieval.enterprise_project_role_rescue import terms
from experiments.retrieval.enterprise_qualification_replay import (
    interpretations, load_scope, score_profile, support,
)
from experiments.retrieval.enterprise_routed_review import digest, rows, write

WIDE_PEOPLE = 20
WIDE_EXCERPTS = 8


def widen(hit, initial, criteria, sources, owned):
    inventory = dict(owned.get(hit["id"], {}))
    for key in prefixes(hit, sources, 800):
        source, start, end = key
        inventory.setdefault(key, terms(sources[source]["text"][start:end]))
    by_condition = [
        sorted(inventory, key=lambda key: (-len(words & inventory[key]), key))
        for words in criteria
    ]
    result = dict.fromkeys(initial)
    for matches in zip(*by_condition, strict=True):
        for reference in matches:
            result.setdefault(reference)
            if len(result) == WIDE_EXCERPTS:
                return tuple(result)
    return tuple(result)


def verify_all(selections, criteria, sources, verifier):
    from experiments.retrieval.enterprise_first_query import TASK, VERIFY_BATCH

    jobs, bank = [], {}
    for owner, spans in selections.items():
        for ref in spans:
            source, start, end = ref
            item = sources[source]
            if item["expert_id"] != owner or not 0 <= start < end <= len(item["text"]):
                raise ValueError("Invalid source ownership or offsets")
            jobs.extend((owner, ref, i, (criterion, item["text"][start:end]))
                        for i, criterion in enumerate(criteria))
    for start in range(0, len(jobs), VERIFY_BATCH):
        batch = jobs[start:start + VERIFY_BATCH]
        scores, clipped = verifier.score_pairs([entry[3] for entry in batch], task=TASK)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Unexpected clipped or nonfinite diagnostic verifier result")
        for (owner, ref, condition, _), score in zip(batch, scores, strict=True):
            bank[owner, ref, condition] = score
    return bank


def evidence_summary(ordered, candidates, bank, conditions):
    cases = Counter(experts=0, checked_excerpts=0, positives_topic=0,
                    positives_all_qualifications=0, positives_all_conditions=0)
    issued = []
    for owner in ordered:
        refs = candidates[owner]
        cases.update(experts=1, checked_excerpts=len(refs))
        best = [max(refs, key=lambda ref: bank[owner, ref, i], default=None)
                for i in range(conditions)]
        flags = [ref is not None and bank[owner, ref, i] > 0 for i, ref in enumerate(best)]
        cases.update(positives_topic=int(flags[0]),
                     positives_all_qualifications=int(all(flags[1:])),
                     positives_all_conditions=int(all(flags)))
        if all(flags):
            issued.append({"expert": owner, "citations": list(dict.fromkeys(best))})
    return dict(cases), issued


def scope_axes(spans, votes):
    original = [tuple(span) for span in spans]
    if sorted(original) != sorted(span for span, (selected, _) in votes.items() if selected):
        raise ValueError("Scope comparison does not match the original qualifications")
    alternative = [span for span, (_, positive) in votes.items() if positive]
    added = sorted(set(alternative) - set(original))
    index = {span: i for i, span in enumerate(original + added, start=1)}
    return added, interpretations([index[span] for span in original],
                                  [index[span] for span in alternative])


def count_interpretations(table, people, excerpts, bank, axes):
    profiles = [score_profile(owner, excerpts[owner], bank, 1 + len(axes["union"]))
                for owner in people]
    for name, conditions in axes.items():
        accepted = sum(support(profile, conditions) for profile in profiles)
        table[name].update(query_has_result=int(bool(accepted)), provisional_people=accepted)


def run(args):
    import torch

    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, EMBED_TASK
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.sepef_first_query import lexical_index

    roles.require_cuda(args.output, args.private)
    anchor = json.loads(args.reference.read_bytes())
    if (digest(args.trace) != anchor["trace_sha256"] or digest(args.queries) != DEVELOPMENT
            or digest(args.corpus) != SOURCE_INVENTORY):
        raise ValueError("Changed pinned development inputs")
    requests = {r["id"]: r["text"] for r in rows(args.queries)}
    scope = None
    if args.scope:
        scope_trace, scope_aggregate = args.scope
        scope = load_scope(anchor, json.loads(scope_aggregate.read_bytes()), scope_trace, requests)
    original = json.loads(args.trace.read_bytes())["observations"]
    fixed = [r for r in original if r["arms"]["hybrid_prefixes"]["qualifications"]]
    if len(fixed) != anchor["multi_condition"] or any(r["arms"]["hybrid_prefixes"]["issued"] for r in fixed):
        raise ValueError("Changed fixed multi-condition panel")
    if scope is not None and set(scope) != {row["id"] for row in fixed}:
        raise ValueError("Scope comparison and headroom use different requests")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {r["id"]: r for r in rows(args.corpus)}
    ids = list(sources)
    owners = [r["expert_id"] for r in sources.values()]
    model_path = args.models / "qwen3-embedding-0.6b"
    revision = verify_checkpoint(model_path)["revision"]
    metadata = json.loads(args.vector_meta.read_bytes())
    if (metadata["corpus_sha256"], metadata["checkpoint"], metadata["vectors_sha256"], metadata["rows"]) != (
            SOURCE_INVENTORY, revision, digest(args.vectors), len(sources)):
        raise ValueError("Vector cache changed")
    encoder = QwenTorchEmbedding(model_path)
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(np.load(args.vectors, allow_pickle=False),
                                                     ids, [[owner] for owner in owners]))
    verifier = Reranker(args.models / "qwen3-reranker-0.6b")
    if anchor["models"]["embedding"] != revision or anchor["models"]["verifier"] != verifier.provenance["revision"]:
        raise ValueError("Changed model revisions")
    windows, _ = roles.load_role_index(args.role_windows, SOURCE_INVENTORY, sources)
    owned = prepare_owned(windows)

    results = {}
    totals = defaultdict(Counter)
    crossed = defaultdict(lambda: defaultdict(Counter))
    with TemporaryDirectory(dir=args.scratch) as root:
        lexical = lexical_index([r["text"] for r in sources.values()], Path(root))
        for n, request in enumerate(fixed[::(-1 if args.order_offset else 1)], 1):
            key = request["id"]
            text = requests[key]
            baseline = request["arms"]["hybrid_prefixes"]
            criteria = (text.strip().splitlines()[0][:CRITERION_CHARS],
                        *(text[a:b] for a, b in baseline["qualifications"]))
            if any(len(q) > CRITERION_CHARS for q in criteria):
                raise ValueError("Unexpected overlength criterion in pinned panel")
            vector = encoder.encode([f"Instruct: {EMBED_TASK}\nQuery: {text}"])[0]
            narrow = hybrid.hybrid_hits(text, vector, index, lexical, owners, ids, WIDE_PEOPLE, 2)
            broad = hybrid.hybrid_hits(text, vector, index, lexical, owners, ids, WIDE_PEOPLE, WIDE_EXCERPTS)
            order = [hit["id"] for hit in narrow]
            if order[:5] != baseline["shortlist"] or [hit["id"] for hit in broad] != order:
                raise ValueError("Changed ordered people pool")
            initial = {hit["id"]: prefixes(hit, sources, 800) for hit in narrow}
            if [list(map(list, initial[owner])) for owner in order[:5]] != baseline["plan"]:
                raise ValueError("Changed first two source excerpts")
            criterion_terms = tuple(map(terms, criteria))
            expanded = {hit["id"]: widen(hit, initial[hit["id"]], criterion_terms, sources, owned)
                        for hit in broad}
            # Preserve the original prefix batch layout. New excerpts are
            # scored separately and reused in every counterfactual panel.
            first = {owner: initial[owner] for owner in order[:5]}
            original_bank = verify_all(first, criteria, sources, verifier)
            extra = {owner: tuple(ref for ref in refs if ref not in first.get(owner, ()))
                     for owner, refs in expanded.items()}
            bank = original_bank | verify_all(extra, criteria, sources, verifier)
            if scope is not None:
                added, axes = scope_axes(baseline["qualifications"], scope[key])
                additions = tuple(text[a:b] for a, b in added)
                if any(len(q) > CRITERION_CHARS for q in additions):
                    raise ValueError("Unexpected overlength alternate condition")
                scope_bank = bank | {
                    (owner, ref, len(criteria) + i): margin
                    for (owner, ref, i), margin in verify_all(expanded, additions, sources, verifier).items()
                }
            private = {}
            for arm, people, excerpts in (
                ("top5_two", order[:5], initial),
                ("top20_two", order, initial),
                ("top5_expanded", order[:5], expanded),
                ("top20_expanded", order, expanded),
                ("rank6_to_20_expanded", order[5:], expanded),
            ):
                stats, issued = evidence_summary(people, excerpts, bank, len(criteria))
                totals[arm].update(stats)
                totals[arm].update(queries=1, queries_with_provisional=int(bool(issued)),
                                   queries_with_topic=int(bool(stats["positives_topic"])),
                                   queries_with_all_qualifications=int(bool(stats["positives_all_qualifications"])))
                private[arm] = issued
                if scope is not None:
                    count_interpretations(crossed[arm], people, excerpts, scope_bank, axes)
            if private["top5_two"]:
                raise ValueError("Original abstention did not reproduce")
            results[key] = dict(arms=private)
            if n % 5 == 0:
                print(f"Completed {n} multi-condition diagnostics", flush=True)
    if scope is not None:
        for arm, summary in crossed.items():
            original = summary["original"]
            if (original["query_has_result"] != totals[arm]["queries_with_provisional"]
                    or original["provisional_people"] != totals[arm]["positives_all_conditions"]):
                raise ValueError("Crossed original interpretation does not reproduce headroom")
    write(args.private, results)
    diagnostic = ({"crossed": crossed, "scope_aggregate_sha256": digest(scope_aggregate)}
                  if scope is not None else {})
    write(args.output, dict(
        arms=totals, **diagnostic, multi_condition=len(fixed), max_people=WIDE_PEOPLE, max_excerpts=WIDE_EXCERPTS,
        reference_sha256=digest(args.reference), private_sha256=digest(args.private),
        script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "role_windows", "vectors", "vector_meta", "models", "scratch",
                 "trace", "reference", "output", "private"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scope", type=Path, nargs=2, metavar=("SCOPE_TRACE", "SCOPE_AGGREGATE"))
    run(parser.parse_args())
