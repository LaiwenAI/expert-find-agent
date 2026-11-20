"""Bound equal-issuance witness comparisons and freeze only influential review views."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path

from experiments.community.crossvalidated_witness import write
from experiments.community.matched_witness_rescue import digest

FIELDS = ("view_id", "supports", "rationale")
VERDICTS = {"yes", "no", "uncertain"}


def possible(row, labels, trust_community):
    old, new = set(row["old"]), set(row["new"])
    known = set(row["known"]) if trust_community else set()
    unknown = sorted((old | new) - known - labels.keys())
    true = known | {key for key, positive in labels.items() if positive}
    outcomes = {}
    for flags in product((False, True), repeat=len(unknown)):
        positives = true | {key for key, flag in zip(unknown, flags, strict=True) if flag}
        old_hits, new_hits = len(old & positives), len(new & positives)
        outcomes[flags] = new_hits - old_hits, int(new_hits > 0) - int(old_hits > 0)
    influential = set()
    for i, key in enumerate(unknown):
        if any(outcomes[flags] != outcomes[flags[:i] + (True,) + flags[i+1:]]
               for flags in outcomes if not flags[i]):
            influential.add(key)
    return zip(*outcomes.values()), influential


def summarize(rows, labels, *, trust_community=False):
    totals, review = {}, set()
    for row in rows:
        bounds, relevant = possible(row, labels, trust_community)
        report = totals.setdefault(row["site"], Counter())
        for metric, outcomes in zip(("citation", "query"), bounds, strict=True):
            report[f"{metric}_delta_min"] += min(outcomes)
            report[f"{metric}_delta_max"] += max(outcomes)
        report.update(
            queries=1, original_issued=len(row["old"]), explicit_issued=len(row["new"]),
            shared_citations=len(set(row["old"]) & set(row["new"])),
            known_original=len(set(row["old"]) & set(row["known"])),
            known_explicit=len(set(row["new"]) & set(row["known"])),
            influential_views=len(relevant),
        )
        review.update(relevant)
    return {site: dict(value) for site, value in totals.items()}, review


def make_views(site, data, corpus, observed):
    from experiments.community.held_window_routing import REQUEST_CHARS
    from experiments.community.witness_matched_issuance import exact_issuance, fixed_threshold, scorecard

    _, records, _, _, _ = corpus
    original, explicit = observed["original"], observed["explicit"]
    old = fixed_threshold(original, 0)
    new = exact_issuance(explicit, sum(map(len, old)))
    rows, views = [], {}
    ids = sorted(map(int, data["documents"]))
    for owner_query, (_, gold), a, b in zip(data["queries"], original, old, new, strict=True):
        keys = {}
        for i in a | b:
            record = records[i]
            signature = f"{site}:{owner_query['qid']}:{ids[i]}"
            key = hashlib.sha256(signature.encode()).hexdigest()[:16]
            if key in views:
                raise ValueError("Duplicate review view")
            keys[i] = key
            views[key] = dict(
                query=f"{owner_query['title']}\n{owner_query['query']}"[:REQUEST_CHARS],
                answer=f"{record['question']}\n{record['answer'][:850]}",
            )
        rows.append(dict(
            site=site, old=sorted(keys[d] for d in a), new=sorted(keys[d] for d in b),
            known=sorted(keys[d] for d in gold & (a | b)),
        ))
    return rows, views, (scorecard(original, old), scorecard(explicit, new))


def save_packet(directory, rows, views, relevant):
    directory.mkdir(mode=0o700)
    # A stable pseudorandom order hides query, method, site, and document IDs.
    ordered = sorted(relevant)
    write(directory / "private-key.json", dict(rows=rows, ids=ordered))
    instructions = [
        "# Blind answer support review",
        "",
        "Judge the displayed answer text against the displayed request. The first answer line is a",
        "context title. Related topics or a matching title alone do not demonstrate support.",
        "",
        "In your CSV, use yes for direct support, no for clear lack of support,",
        "or uncertain for insufficient context. Enter a short rationale for every judgment.",
        "Use only the displayed text. Complete your CSV independently.",
        "",
    ]
    for key in ordered:
        instructions.extend((f"## {key}", "", "Request", ""))
        instructions.extend(f"    {line}" for line in views[key]["query"].splitlines())
        instructions.extend(("", "Answer view", ""))
        instructions.extend(f"    {line}" for line in views[key]["answer"].splitlines())
        instructions.append("")
    (directory / "review-views.md").write_text("\n".join(instructions), encoding="utf-8")
    for name in ("a", "b"):
        with (directory / f"reviewer-{name}.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows((key, "", "") for key in ordered)


def generate(args):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.community.held_window_routing import independent, index, load
    from experiments.community.witness_matched_issuance import cross_site_overlap, observe
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or args.output.exists() or args.packet.exists():
        raise ValueError("Require Slurm and new outputs")
    previous = json.loads(args.reference.read_bytes())
    for field in ("dev_manifest", "stats_manifest", "math_manifest"):
        if previous["manifests"][field] != digest(getattr(args, field)):
            raise ValueError("Changed benchmark manifest")
    dev = load(args.dev, args.dev_manifest)
    stats, _ = independent(dev, load(args.stats, args.stats_manifest))
    math, _ = cross_site_overlap((dev, stats), load(args.math, args.math_manifest))
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    if previous["revisions"] != dict(embedding=verify_checkpoint(args.embedding)["revision"],
                                     reranker=reader.provenance["revision"]):
        raise ValueError("Changed model checkpoint")
    rows, views = [], {}
    for site, data in (("crossvalidated", stats), ("mathoverflow", math)):
        corpus = index(data, encoder)
        raw, _ = observe(data, corpus, encoder, reader)
        selected, evidence, scores = make_views(site, data, corpus, raw)
        modes = ("original", "explicit_exact_budget")
        for arm, observed in zip(modes, scores, strict=True):
            expected = previous["reports"][site]["arms"][arm]
            if any(expected[key] != value for key, value in observed.items()):
                raise ValueError("Published reference counts do not reproduce")
        rows.extend(selected)
        if views.keys() & evidence.keys():
            raise ValueError("Overlapping blind view handles")
        views.update(evidence)
    conditional, community_affected = summarize(rows, {}, trust_community=True)
    bounds, independent_affected = summarize(rows, {})
    affected = independent_affected | community_affected
    save_packet(args.packet, rows, views, affected)
    write(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], code_sha256=digest(Path(__file__)), reference_sha256=digest(args.reference),
        packet_key_sha256=digest(args.packet / "private-key.json"), views_sha256=digest(args.packet / "review-views.md"),
        bounds=bounds, conditional_community_bounds=conditional, influential_views=len(affected),
    ))


def review_answers(path, expected):
    with path.open(encoding="utf-8-sig", newline="") as f:
        table = csv.DictReader(f)
        if table.fieldnames != list(FIELDS):
            raise ValueError("Unexpected review columns")
        answers = {}
        for row in table:
            key, verdict, rationale = (row[name].strip() for name in FIELDS)
            if key in answers or key not in expected or verdict not in VERDICTS or not rationale:
                raise ValueError("Duplicate ID, changed ID, missing rationale or invalid judgment")
            answers[key] = verdict
    if set(answers) != expected:
        raise ValueError("Reviewer missed assigned views")
    return answers


def adjudicated(args):
    anchor = json.loads(args.summary.read_bytes())
    if (digest(args.packet / "private-key.json") != anchor["packet_key_sha256"]
            or digest(args.packet / "review-views.md") != anchor["views_sha256"]):
        raise ValueError("Changed review views or private assignment")
    if args.a.resolve() == args.b.resolve():
        raise ValueError("Require two separate reviewer files")
    record = json.loads((args.packet / "private-key.json").read_bytes())
    expected = set(record["ids"])
    a, b = (review_answers(path, expected) for path in (args.a, args.b))
    labels = {key: a[key] == "yes" for key in expected if a[key] == b[key] and a[key] != "uncertain"}
    needs_adjudication = {key for key in expected if a[key] != b[key]}
    if args.arbiter:
        if args.arbiter.resolve() in (args.a.resolve(), args.b.resolve()):
            raise ValueError("Require a separate adjudicator file")
        vote = review_answers(args.arbiter, needs_adjudication)
        labels.update({key: vote[key] == "yes" for key in needs_adjudication if vote[key] != "uncertain"})
    bounds, affected = summarize(record["rows"], labels)
    write(args.output, dict(
        bounds=bounds, resolved=len(labels), influential_views_unresolved=len(affected),
        disagreements=len(needs_adjudication), assigned=len(expected),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="stage", required=True)
    freeze = stages.add_parser("freeze")
    for name in ("dev", "dev_manifest", "stats", "stats_manifest", "math", "math_manifest",
                 "reference", "embedding", "reranker", "packet", "output"):
        freeze.add_argument(name, type=Path)
    reconcile = stages.add_parser("score")
    for name in ("packet", "summary", "a", "b", "output"):
        reconcile.add_argument(name, type=Path)
    reconcile.add_argument("--arbiter", type=Path)
    args = cli.parse_args()
    (generate if args.stage == "freeze" else adjudicated)(args)
