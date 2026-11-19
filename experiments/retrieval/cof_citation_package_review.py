"""Build and validate a blind review of real returned citation packages."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path

from experiments.retrieval.cof_rank_disagreement_review import sha, write_json, write_rows


ARMS = ("mean_max", "specific_citations")
LIMITS = {"changed": 16, "one_sided": 8, "same_supported": 8}
RELATIONS = {
    "content": {"supports", "contradicts", "insufficient"},
    "contribution": {"explicit", "contradicts", "insufficient"},
}
EDITABLE = {"reviewer", "content", "contribution", "reason", "quotation",
            "publisher_url", "contribution_quote", "identity_sources"}
SOURCE = "cof-citation-package-review-v1"


def key(*values):
    return sha256(json.dumps((SOURCE, *values), ensure_ascii=False).encode()).hexdigest()


def candidates(rows):
    selected = defaultdict(list)
    for query, left, right in rows:
        if list(left) != list(right):
            raise ValueError("Citation comparison requires identical reviewer order")
        for owner in left:
            x, y = left[owner], right[owner]
            if x and y:
                typ = "same_supported" if set(x) == set(y) else "changed"
            elif x or y:
                typ = "one_sided"
            else:
                continue
            selected[typ].append((query, owner, tuple(x), tuple(y)))
    return selected

def population_bounds(manifest, reference):
    """Conservatively bound the full-panel delta before any reviews."""
    n = manifest["top5_slots"]
    left, right = (round(reference["arms"][arm]["citation_coverage"] * n) for arm in ARMS)
    changed, one = (manifest["population"][k] for k in ("changed", "one_sided"))
    if (one + left - right) % 2:
        raise ValueError("Inconsistent one-sided package counts")
    left_only = (one + left - right) // 2
    right_only = one - left_only
    if min(left_only, right_only) < 0:
        raise ValueError("One-sided package counts cannot be negative")
    return dict(left_only=left_only, right_only=right_only,
                clean_delta_bounds=[-(changed + left_only) / n, (changed + right_only) / n])


def export(dataset, result_path, embedding, reranker, output):
    import torch
    from experiments.retrieval import cof_global_profile_pooling as base
    from experiments.retrieval import cof_ownership_specificity as search

    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("Require a new packet in Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = json.loads(result_path.read_bytes())
    if reference["dataset"] != dataset or reference["script_sha256"] != base.digest(Path(search.__file__)):
        raise ValueError("Unexpected reference")
    metadata = base.verify_checkpoint(embedding)
    _, papers, _, owners, ids, _, labels = base.read_data(dataset, metadata)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    index = search.OwnershipIndex(owners)
    encoder = base.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = base.Reranker(reranker)
    queries = base.select(dataset, ids, texts)
    if (sha256("\n".join(queries).encode()).hexdigest() != reference["cohort_sha256"]
            or verifier.provenance["revision"] != reference["reranker_revision"]):
        raise ValueError("Changed input cohort or verifier")
    result = []
    counts = {arm: Counter() for arm in ARMS}
    for q in queries:
        by_arm = {}
        for arm in ARMS:
            row = base.request(index, encoder, verifier, texts, texts[q], labels[q], arm)
            citations = dict(row["returned_citations"])
            order = [owner for owner, _ in row["ranked_sources"][:base.TOP]]
            by_arm[arm] = {owner: citations.get(owner, ()) for owner in order}
            counts[arm]["coverage"] += sum(bool(cs) for cs in citations.values())
            counts[arm]["positive"] += sum(bool(cs) and labels[q].get(owner) == 1 for owner, cs in citations.items())
        result.append((q, *(by_arm[arm] for arm in ARMS)))
    for arm, count in counts.items():
        if any(count[f] != round(reference["arms"][arm][metric] * len(queries) * base.TOP)
               for f, metric in (("coverage", "citation_coverage"), ("positive", "cited_p5"))):
            raise ValueError("Return counts differ from the published live experiment")

    buckets = candidates(result)
    review, mapping = {}, []
    for category, limit in LIMITS.items():
        cohort = sorted(buckets[category], key=lambda row: key(dataset, category, *row[:2]))[:limit]
        for q, owner, first, second in cohort:
            cases = {pid: key(dataset, q, owner, pid)[:20] for pid in set(first) | set(second)}
            owner_alias = key(dataset, q, owner)[:8]
            profile_papers = [owners.source_ids[i] for i in owners.owners[owner]]
            profile_papers.sort(key=lambda pid: (index.source_degree[pid], key(dataset, owner, pid)))
            anchors = [papers[pid]["title"] for pid in profile_papers if pid not in cases][:3]
            for pid, case in cases.items():
                if case in review:
                    raise ValueError("Duplicate blind cue")
                review[case] = dict(
                    case=case, candidate=owner_alias, query=texts[q], citation=texts[pid], profile_titles=anchors,
                    reviewer=None, content=None, contribution=None, reason=None, quotation=None,
                    publisher_url=None, contribution_quote=None, identity_sources=None,
                )
            mapping.append(dict(category=category, query=q, owner=owner,
                                first=[cases[pid] for pid in first], second=[cases[pid] for pid in second]))
    output.mkdir(parents=True)
    private = output / "custodian"
    private.mkdir()
    form = output / "review.jsonl"
    write_rows(form, [review[c] for c in sorted(review, key=lambda c: key("order", c))])
    pair_map = private / "pairs.json"
    write_json(pair_map, mapping)
    summary = dict(dataset=dataset, source_job=reference["job"], packet_job=os.environ["SLURM_JOB_ID"],
                   reference_sha256=sha(result_path), queries=len(queries), top5_slots=len(queries) * base.TOP,
                   population={label: len(buckets[label]) for label in LIMITS},
                   sampled={label: sum(x["category"] == label for x in mapping) for label in LIMITS},
                   citation_cases=len(review), human_reviews=0, form_sha256=sha(form), mapping_sha256=sha(pair_map))
    summary["full_panel_before_review"] = population_bounds(summary, reference)
    write_json(output / "manifest.json", summary)
    (output / "REVIEW.md").write_text(
        "# Independent citation assessment\n\n"
        "Assess each citation against the displayed research query. Mark content as supports, contradicts, "
        "or insufficient. Assess contribution separately for the visible candidate alias. For explicit or "
        "contradictory contribution, provide a public publisher URL for the same paper and a verbatim excerpt "
        "that connects the candidate to the relevant work. Profile titles are separate from the citation. "
        "Use their public bylines or linked persistent researcher identifiers to establish a unique identity. "
        "Record at least two identity source URLs. Use insufficient if these sources leave several plausible "
        "people. Profile membership alone is insufficient. A definite content decision requires an exact "
        "quotation from the displayed citation. Use contribution_quote for the publisher evidence and "
        "record the identity and capability reasoning in reason. "
        "Use a stable reviewer pseudonym and leave incomplete rows blank. Do not consult the custodian folder.\n"
    )
    print(json.dumps(summary, indent=2))


def read_form(path):
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["case"] in result:
                raise ValueError("Duplicate case")
            result[row["case"]] = row
    return result


def checked(template, form):
    rows = read_form(form)
    if rows.keys() != template.keys():
        raise ValueError("Case inventory changed")
    identities = set()
    for case, row in rows.items():
        original = template[case]
        if row.keys() != original.keys() or any(row[k] != original[k] for k in original.keys() - EDITABLE):
            raise ValueError("Changed case content")
        values = [row[axis] for axis in RELATIONS]
        if all(value is None for value in values):
            if any(row[k] is not None for k in EDITABLE):
                raise ValueError("Keep incomplete rows empty")
            continue
        if (any(row[axis] not in labels for axis, labels in RELATIONS.items())
                or any(not isinstance(row[k], str) or not row[k].strip() for k in ("reviewer", "reason"))):
            raise ValueError("Incomplete judgment")
        identities.add(row["reviewer"].strip())
        excerpt = row["quotation"]
        quotes = [q.strip() for q in excerpt.splitlines() if q.strip()] if isinstance(excerpt, str) else []
        if row["content"] != "insufficient" and not any(q in original["citation"] for q in quotes):
            raise ValueError("Content judgment requires a quotation from the displayed citation")
        if row["contribution"] != "insufficient":
            links = row["identity_sources"]
            publisher = row["publisher_url"]
            section = row["contribution_quote"]
            if (not isinstance(section, str) or not section.strip() or not isinstance(publisher, str)
                    or not publisher.startswith("https://") or not isinstance(links, list)
                    or any(not isinstance(url, str) or not url.startswith("https://") for url in links)
                    or len(set(links) - {publisher}) < 2):
                raise ValueError("Require a publisher quote and distinct identity evidence URLs")
    if len(identities) > 1:
        raise ValueError("Use a separate copy for each reviewer")
    return rows, identities


def clean_difference(first, second, outcomes):
    first, second = set(first), set(second)
    if first == second:
        return 0, 0
    unknown = sorted(case for case in first | second if outcomes[case] is None)
    differences = []
    for assignment in product((0, 1), repeat=len(unknown)):
        resolved = {**outcomes, **dict(zip(unknown, assignment, strict=True))}
        clean = lambda cases: int(bool(cases) and all(resolved[c] == 1 for c in cases))
        differences.append(clean(second) - clean(first))
    return min(differences), max(differences)


def compare(packet, first, second, output):
    if output.exists():
        raise ValueError("Choose a fresh aggregate output")
    manifest = json.loads((packet / "manifest.json").read_bytes())
    form, pair_map = packet / "review.jsonl", packet / "custodian" / "pairs.json"
    if sha(form) != manifest["form_sha256"] or sha(pair_map) != manifest["mapping_sha256"]:
        raise ValueError("Packet changed")
    template = read_form(form)
    a, left_ids = checked(template, first)
    b, right_ids = checked(template, second)
    if left_ids & right_ids:
        raise ValueError("Use independently attributed reviews")
    outcomes, states = {}, Counter()
    axes = {axis: Counter() for axis in RELATIONS}
    for case in template:
        for axis, counter in axes.items():
            first_value, second_value = a[case][axis], b[case][axis]
            status = ("missing" if None in (first_value, second_value)
                      else first_value if first_value == second_value else "disputed")
            counter[status] += 1
        x, y = [(r[case]["content"], r[case]["contribution"]) for r in (a, b)]
        if None in x or None in y:
            state, label = "missing", None
        elif x != y:
            state, label = "disputed", None
        elif "contradicts" in x:
            state, label = "contradicted", 0
        elif "insufficient" in x:
            state, label = "insufficient", None
        else:
            state, label = "supported", 1
        states[state] += 1
        outcomes[case] = label
    totals, strata = Counter(), defaultdict(Counter)
    for pair in json.loads(pair_map.read_bytes()):
        x, y = pair["first"], pair["second"]
        lo, hi = clean_difference(x, y, outcomes)
        values = dict(sampled_slots=1, delta_lower_slots=lo, delta_upper_slots=hi)
        totals.update(values)
        strata[pair["category"]].update(values)

    def bounds(counts):
        n = counts["sampled_slots"]
        return [counts["delta_lower_slots"]/n, counts["delta_upper_slots"]/n] if n else [0, 0]

    write_json(output, dict(
        dataset=manifest["dataset"], paired_cases=dict(states),
        agreements_by_axis={name: dict(counts) for name, counts in axes.items()},
        sampled_slots=totals["sampled_slots"], clean_delta_bounds=bounds(totals),
        strata={name: dict(sampled_slots=c["sampled_slots"], clean_delta_bounds=bounds(c))
                for name, c in sorted(strata.items())},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    setup = modes.add_parser("export")
    setup.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("result_path", "embedding", "reranker", "output"):
        setup.add_argument(name, type=Path)
    finish = modes.add_parser("compare")
    for name in ("packet", "first", "second", "output"):
        finish.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    {"export": export, "compare": compare}[args.pop("mode")](**args)
