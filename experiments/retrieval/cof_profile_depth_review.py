"""Blind review of the evidence gaps exposed by the fixed CoF depth audit."""

import argparse
from collections import Counter, defaultdict
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unicodedata import normalize

from experiments.retrieval.cof_citation_package_review import EDITABLE, checked, key, read_form
from experiments.retrieval.cof_rank_disagreement_review import sha, write_json, write_rows

CONTROL_LIMIT = 6
EXTRA = {"full_text_content", "full_text_url", "full_text_quote"}
FULL_TEXT = {"supports", "contradicts", "insufficient", "unavailable"}
REPLAY_KEYS = (
    "dataset", "queries", "reviewers", "sources", "cohort_sha256", "labels_sha256",
    "groups", "scored_pairs", "truncated_pairs", "embedding_revision", "reranker_revision",
    "score_batch_size", "score_sequence_cap",
)


def normalized_text(text):
    return " ".join(re.findall(r"[^\W_]+", normalize("NFKC", text).casefold()))


def fingerprints(text, source):
    title = normalized_text(text.partition("\n")[0])
    full = normalized_text(text)
    return (title or full or source), (full or source)


def source_counts(sources, texts, margins):
    entries = [(position, fingerprints(texts[doc], doc)) for position, doc in enumerate(sources, 1)
               if margins[doc] >= 0]
    counts = Counter(profiles=1)
    for column, name in enumerate(("normalized_title", "normalized_text")):
        for field, depth in (("3", 3), ("6", 6), ("12", 12), ("catalog", len(sources))):
            distinct = {pair[column] for position, pair in entries if position <= depth}
            counts[f"two_by_{name}_at_{field}"] = int(len(distinct) >= 2)
        counts[f"one_by_{name}_in_catalog"] = int(len({pair[column] for _, pair in entries}) == 1)
    return counts


def classify(sources, margins):
    accepted = [doc for doc in sources if margins[doc] >= 0]
    if sum(margins[doc] >= 0 for doc in sources[:3]) >= 2:
        stratum = "supported_at_three"
    elif len(accepted) >= 2:
        stratum = "rescued_beyond_three"
    elif len(accepted) == 1:
        stratum = "only_one_in_catalog"
    else:
        stratum = "none_in_catalog"
    selected = list(dict.fromkeys([*sources[:3], *accepted[:2]]))
    if stratum in {"only_one_in_catalog", "none_in_catalog"}:
        alternate = max((doc for doc in sources[3:] if margins[doc] < 0),
                        key=lambda doc: margins[doc], default=None)
        if alternate is not None and alternate not in selected:
            selected.append(alternate)
    return stratum, selected


def export(dataset, reference, embedding, reranker, output):
    from experiments.retrieval import cof_full_profile_support_audit as audit

    if output.exists() or not output.parent.is_dir():
        raise ValueError("Use a fresh packet path in an existing directory")
    published = json.loads(reference.read_bytes())
    if published["dataset"] != dataset:
        raise ValueError("Reference names a different panel")
    groups = defaultdict(list)
    deduplicated = Counter()

    def collect(query, texts, owner, sources, margins):
        stratum, selected = classify(sources, margins)
        blind_owner = key("profile-depth", dataset, query, owner)[:12]
        deduplicated.update(source_counts(sources, texts, margins))
        excluded_titles = {fingerprints(texts[doc], doc)[0] for doc in selected}
        titles = []
        for doc in sources:
            identity_key, _ = fingerprints(texts[doc], doc)
            if identity_key not in excluded_titles:
                excluded_titles.add(identity_key)
                titles.append(texts[doc].partition("\n")[0])
            if len(titles) == 3:
                break
        cases, metadata = [], []
        for doc in selected:
            case = key("profile-depth", dataset, query, owner, doc)[:20]
            cases.append(dict(case=case, candidate=blind_owner, query=texts[query],
                              citation=texts[doc], profile_titles=titles,
                              **dict.fromkeys(EDITABLE | EXTRA)))
            metadata.append(dict(case=case, source=doc, prefix=doc in sources[:3],
                                 model_accepted=margins[doc] >= 0))
        groups[stratum].append((blind_owner, cases, dict(
            stratum=stratum, query=query, owner=owner, sources=metadata,
        )))

    with TemporaryDirectory(prefix="cof-depth-replay-", dir=output.parent) as directory:
        replay = Path(directory) / "replay.json"
        audit.run(SimpleNamespace(dataset=dataset, embedding=embedding, reranker=reranker, output=replay),
                  observer=collect)
        fresh = json.loads(replay.read_bytes())
        if any(fresh[field] != published[field] for field in REPLAY_KEYS):
            raise ValueError("Replay does not match the published source-depth audit")
    cohort = published["groups"]["ranks_1_to_5"]
    expected = {
        "supported_at_three": cohort["two_accepted_at_3"],
        "rescued_beyond_three": cohort["two_accepted_in_catalog"] - cohort["two_accepted_at_3"],
        "only_one_in_catalog": cohort["one_accepted_in_catalog"],
        "none_in_catalog": cohort["zero_accepted_in_catalog"],
    }
    if any(len(groups[stratum]) != size for stratum, size in expected.items()):
        raise ValueError("Unexpected candidate stratum sizes")

    forms, mapping, sizes = [], [], {}
    for stratum in expected:
        cases = sorted(groups[stratum], key=lambda pair: key("depth-control", pair[0]))
        if stratum == "supported_at_three":
            cases = cases[:CONTROL_LIMIT]
        sizes[stratum] = len(cases)
        for _, rows, metadata in cases:
            forms.extend(rows)
            mapping.append(metadata)
    if len({row["case"] for row in forms}) != len(forms):
        raise ValueError("Blind case identifiers must be unique")
    output.mkdir(mode=0o700)
    private = output / "custodian"
    private.mkdir(mode=0o700)
    form, case_map = output / "review.jsonl", private / "cases.json"
    write_rows(form, sorted(forms, key=lambda row: key("depth-order", row["case"])))
    write_json(case_map, mapping)
    (output / "REVIEW.md").write_text(
        "# Independent evidence review\n\n"
        "Work from a private copy of review.jsonl. Keep its case IDs and input fields unchanged. "
        "Use one stable reviewer pseudonym per copy. Leave an unreviewed row entirely blank.\n\n"
        "First read the displayed citation. Mark content as supports, contradicts, or insufficient "
        "for the stated query. Give an exact displayed quotation for a definite result. Then locate "
        "the same source's full text. Record full_text_content as supports, contradicts, insufficient, "
        "or unavailable. Supply its public URL when accessible and quote the text for a definite result.\n\n"
        "Assess contribution separately. Use profile titles to resolve the candidate's identity. "
        "When full text or its contribution statement explicitly links the candidate to the relevant "
        "work, mark explicit. Mark contradicts only for direct conflicting evidence. For either label, "
        "provide the publisher URL, an exact contribution quote and at least two separate identity "
        "source URLs. Otherwise use insufficient. Authorship alone does not establish a specific contribution. "
        "Note any cited records that resolve to the same publication. Record the rationale in reason. Do not consult the custodian map.\n"
    )
    write_json(output / "manifest.json", dict(
        dataset=dataset, reference_sha256=sha(reference), audit_runner_sha256=fresh["script_sha256"],
        packet_script_sha256=sha(Path(__file__)), packet_job=os.environ["SLURM_JOB_ID"],
        pool_sizes=expected, sampled_profiles=sizes, citation_cases=len(forms),
        distinct_source_audit=deduplicated,
        no_profile_anchor_cases=sum(not row["profile_titles"] for row in forms),
        form_sha256=sha(form), mapping_sha256=sha(case_map), human_reviews=0,
    ))


def full_text_rows(template, form):
    rows, identities = checked(template, form, editable=EDITABLE | EXTRA)
    for row in rows.values():
        if row["content"] is None:
            continue
        content = row["full_text_content"]
        if content not in FULL_TEXT:
            raise ValueError("Record the independent full-text assessment")
        if content != "unavailable":
            url = row["full_text_url"]
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError("Accessible full text requires a public source URL")
        if content in {"supports", "contradicts"}:
            quote = row["full_text_quote"]
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("Definite full-text content requires a direct quote")
    return rows, identities


def compare(packet, first, second, output):
    if output.exists():
        raise ValueError("Choose a fresh review summary path")
    manifest = json.loads((packet / "manifest.json").read_bytes())
    form = packet / "review.jsonl"
    map_path = packet / "custodian" / "cases.json"
    if sha(form) != manifest["form_sha256"] or sha(map_path) != manifest["mapping_sha256"]:
        raise ValueError("Unchanged template and custodian map required")
    mapping = json.loads(map_path.read_bytes())
    template = read_form(form)
    left, left_ids = full_text_rows(template, first)
    right, right_ids = full_text_rows(template, second)
    if left_ids & right_ids:
        raise ValueError("Use independently attributed review forms")
    axes = ("content", "full_text_content", "contribution")
    tables = defaultdict(Counter)
    for profile in mapping:
        for source in profile["sources"]:
            case = source["case"]
            cell = (profile["stratum"], "first_three" if source["prefix"] else "deeper",
                    "model_accept" if source["model_accepted"] else "model_reject")
            a, b = left[case], right[case]
            row = tables[cell]
            row["sources"] += 1
            states = []
            for axis in axes:
                first_label, second_label = a[axis], b[axis]
                state = ("missing" if None in (first_label, second_label)
                         else first_label if first_label == second_label else "disputed")
                states.append(state)
                row[f"{axis}_{state}"] += 1
            if states == ["supports", "supports", "explicit"]:
                row["visible_and_full_text_support_with_contribution"] += 1
            if states == ["insufficient", "supports", "explicit"]:
                row["full_text_only_support_with_contribution"] += 1
    result = []
    for (stratum, depth, verdict), counts in sorted(tables.items()):
        result.append(dict(stratum=stratum, depth=depth, verifier=verdict, **counts))
    write_json(output, dict(dataset=manifest["dataset"], packet_sha256=sha(packet / "manifest.json"),
                            citation_cases=len(template), tables=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("export")
    build.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for field in ("reference", "embedding", "reranker", "output"):
        build.add_argument(field, type=Path)
    score = modes.add_parser("compare")
    for field in ("packet", "first", "second", "output"):
        score.add_argument(field, type=Path)
    options = vars(parser.parse_args())
    method = options.pop("mode")
    (export if method == "export" else compare)(**options)
