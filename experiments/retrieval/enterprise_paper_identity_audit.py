"""Audit offline authorship hints in the frozen enterprise development catalog."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from pypinyin import Style, lazy_pinyin


PAPER_DOI = re.compile(r"10\.\d{4,9}/[^\s]+", re.IGNORECASE)
SCHOLAR_CLUSTER = re.compile(r"\d{8,22}")
HAN = re.compile(r"[\u3400-\u9fff]{2,4}")
AUTHOR_SEPARATOR = re.compile(r"[,，;；、|＆&]|\s+and\s+", re.IGNORECASE)


def digest(path):
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def record_hash(record):
    return sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def key(value):
    n = unicodedata.normalize("NFKD", value or "").casefold()
    return "".join(c for c in n if c.isalnum())


def names(owner):
    exact = key(owner)
    aliases = set()
    if HAN.fullmatch(exact):
        syllables = lazy_pinyin(exact, style=Style.NORMAL)
        aliases = {"".join(syllables), "".join(syllables[1:] + syllables[:1])}
        aliases = {alias for alias in aliases if re.fullmatch(r"[a-z]{4,}", alias)}
    return exact, aliases


def roster(text):
    if not text:
        return set()
    return {name for span in AUTHOR_SEPARATOR.split(text)
            if (name := key(span)) and name not in {"etal"} and len(name) >= 2}


def inventory(dataset, sidecar, raw_path):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    meta = json.loads((sidecar / "manifest.json").read_bytes())
    for base, filename, expected in (
        (dataset, "corpus.jsonl", manifest["files_sha256"]["corpus.jsonl"]),
        (dataset, "experts.jsonl", manifest["files_sha256"]["experts.jsonl"]),
        (sidecar, "papers.jsonl", meta["metadata_sha256"]),
        (raw_path.parent, raw_path.name, meta["source_sha256"]),
    ):
        if digest(base / filename) != expected:
            raise ValueError(f"Changed inventory: {filename}")
    owners = {x["id"]: names(x["name"]) for line in (dataset / "experts.jsonl").open()
              if (x := json.loads(line))["split"] == "development"}
    wanted = {x["id"]: x["expert_id"] for line in (dataset / "corpus.jsonl").open()
              if (x := json.loads(line))["split"] == "development" and x["kind"] == "paper"}
    rows = {}
    with raw_path.open() as raw, (sidecar / "papers.jsonl").open() as source:
        for number, (raw_line, side_line) in enumerate(zip(raw, source, strict=True), 1):
            s = json.loads(side_line)
            if s["source_id"] not in wanted:
                continue
            if s["raw_line"] != number or wanted[s["source_id"]] != s["expert_id"]:
                raise ValueError("Source ordinal or profile owner mismatch")
            r = json.loads(raw_line)
            if record_hash(r) != s["raw_record_sha256"]:
                raise ValueError("Changed raw record")
            doi = (s["doi"] or "").casefold().strip()
            scholar = str(r.get("gs_google_scholar_cites_id") or "").strip()
            pdf = urlsplit(r.get("gs_pdf_link") or "")
            rows[s["source_id"]] = dict(
                owner=s["expert_id"], title=key(r["title"]), authors=roster(s["author_list_text"]),
                doi=doi if PAPER_DOI.fullmatch(doi) else "",
                scholar=scholar if SCHOLAR_CLUSTER.fullmatch(scholar) else "",
                pdf_link=pdf.scheme in {"https", "http"} and bool(pdf.hostname),
                contributor_roles=bool(s["contribution"]["verified_roles"]),
            )
    if rows.keys() != wanted.keys() or set(wanted.values()) - owners.keys():
        raise ValueError("The pinned development paper inventory is incomplete")
    return owners, rows


def identifiers(paper):
    return {(field, paper[field]) for field in ("doi", "scholar") if paper[field]}


def identifier_conflicts(papers):
    labels = defaultdict(set)
    for paper in papers.values():
        for identity in identifiers(paper):
            if paper["title"]:
                labels[identity].add(paper["title"])
    return {identity for identity, observed_titles in labels.items() if len(observed_titles) > 1}


def views(owners, papers):
    name_population = Counter(exact for exact, _ in owners.values() if exact)
    alias_population = Counter(alias for _, aliases in owners.values() for alias in aliases)
    aliases = {owner: {alias for alias in options if alias_population[alias] == 1}
               for owner, (_, options) in owners.items()}
    by_name, owned = defaultdict(set), defaultdict(set)
    statuses = {}
    conflicts = identifier_conflicts(papers)
    for source, paper in papers.items():
        owner = paper["owner"]
        exact, _ = owners[owner]
        authors = paper["authors"]
        exact_match = bool(exact and exact in authors)
        phonetic = authors & aliases[owner]
        tagged = identifiers(paper)
        statuses[source] = dict(
            roster=bool(authors), exact=exact_match,
            unique_exact=exact_match and name_population[exact] == 1,
            phonetic=bool(phonetic),
            local_id=bool(tagged), doi=bool(paper["doi"]), scholar=bool(paper["scholar"]),
            reported_pdf_link=paper["pdf_link"],
            clean_local_id=bool(tagged) and not (tagged & conflicts),
            role=paper["contributor_roles"],
        )
        owned[owner].add(source)
        for author in authors:
            by_name[author].add(source)
    return statuses, owned, by_name, name_population, aliases, conflicts


def analyze(owners, papers, trace):
    statuses, owned, by_name, name_population, aliases, conflicts = views(owners, papers)
    inventory = Counter(paper_rows=len(papers), catalog_owners=len(owners),
                        doi_or_cluster_with_multiple_local_titles=len(conflicts))
    for status in statuses.values():
        inventory.update(status)
    for exact, _ in owners.values():
        inventory["owners_with_ambiguous_exact_name"] += bool(exact and name_population[exact] > 1)
    inventory["owners_with_no_unique_transliteration"] = sum(bool(options) and not aliases[owner]
                                                          for owner, (_, options) in owners.items())
    shortlists = [row["arms"]["hybrid_prefixes"] for row in trace["observations"]]
    if len(shortlists) != 80 or len({row["id"] for row in trace["observations"]}) != 80:
        raise ValueError("Use the frozen development trace")
    candidates = set(owner for arm in shortlists for owner in arm["shortlist"])
    if candidates - owners.keys():
        raise ValueError("Unknown candidate in shortlist")

    metrics = {}
    row_diagnostics = Counter()
    for owner in candidates:
        own = owned[owner]
        exact, _ = owners[owner]
        named = {source for source in own if statuses[source]["unique_exact"]}
        aliased = {source for source in own if statuses[source]["unique_exact"] or statuses[source]["phonetic"]}
        anchor_coauthors = {author for source in named for author in papers[source]["authors"]} - {exact} - aliases[owner]
        prior_titles = {papers[source]["title"] for source in own if papers[source]["title"]}
        prior_identifiers = set().union(*(identifiers(papers[source]) for source in own))
        foreign = set().union(*(by_name[name] for name in aliases[owner] | ({exact} if name_population[exact] == 1 else set())))
        distinct = {source for source in foreign - own
                    if papers[source]["title"] and papers[source]["title"] not in prior_titles
                    and not identifiers(papers[source]) & prior_identifiers}
        corroborated = {source for source in distinct if papers[source]["authors"] & anchor_coauthors}
        with_key = {source for source in corroborated if statuses[source]["clean_local_id"]}
        exact_titles = {papers[source]["title"] for source in named if papers[source]["title"]}
        clean_exact = {papers[source]["title"] for source in named if papers[source]["title"]
                       and statuses[source]["clean_local_id"]}
        metrics[owner] = {
            "has_paper": bool(own), "roster": any(statuses[s]["roster"] for s in own),
            "local_identifier": any(statuses[s]["local_id"] for s in own),
            "local_doi": any(statuses[s]["doi"] for s in own),
            "local_pdf_link": any(statuses[s]["reported_pdf_link"] for s in own),
            "catalog_unique_name": bool(named), "exact_or_unique_transliteration": bool(aliased),
            "two_distinct_titles_with_exact_name": len(exact_titles) >= 2,
            "two_distinct_titles_with_exact_name_and_local_key": len(clean_exact) >= 2,
            "a_foreign_distinct_title_name_hint": bool(distinct),
            "a_foreign_distinct_title_coauthor_hint": bool(corroborated),
            "a_foreign_distinct_title_coauthor_hint_and_local_key": bool(with_key),
            "contribution_role": any(statuses[s]["role"] for s in own | corroborated),
        }
        row_diagnostics.update(foreign_name_candidate_rows=len(distinct), foreign_coauthor_candidate_rows=len(corroborated),
                               foreign_coauthor_and_key_candidate_rows=len(with_key),
                               foreign_rows_excluded_by_title_or_id_screen=len(foreign - own - distinct))

    report = defaultdict(Counter)
    for arm in shortlists:
        chosen = arm["shortlist"]
        if len(chosen) != len(set(chosen)) or len(chosen) != len(arm["plan"]):
            raise ValueError("Invalid shortlist or read plan")
        flags = [metrics[owner] for owner in chosen]
        read_states = []
        for owner, views_read in zip(chosen, arm["plan"], strict=True):
            paper_refs = set()
            for source, _, _ in views_read:
                if source in papers:
                    if papers[source]["owner"] != owner:
                        raise ValueError("Read plan moved evidence between people")
                    paper_refs.add(source)
            read_states.append({
                "selected_a_paper": bool(paper_refs),
                "selected_a_paper_with_name": any(statuses[s]["unique_exact"] or statuses[s]["phonetic"] for s in paper_refs),
                "selected_a_paper_with_local_identifier": any(statuses[s]["local_id"] for s in paper_refs),
                "selected_a_paper_with_local_doi": any(statuses[s]["doi"] for s in paper_refs),
                "selected_a_paper_with_pdf_link": any(statuses[s]["reported_pdf_link"] for s in paper_refs),
                "selected_a_paper_with_contribution_role": any(statuses[s]["role"] for s in paper_refs),
            })
        buckets = ["all"] + (["initial_abstentions"] if not arm["issued"] else [])
        for bucket in buckets:
            dest = report[bucket]
            dest.update(requests=1, slots=len(chosen))
            for key in flags[0]:
                hits = sum(row[key] for row in flags)
                dest.update({key + "_slots": hits, key + "_requests": bool(hits)})
            for key in read_states[0]:
                hits = sum(row[key] for row in read_states)
                dest.update({key + "_slots": hits, key + "_requests": bool(hits)})
    return dict(inventory=dict(inventory), shortlist_owner_diagnostics=dict(row_diagnostics),
                shortlist_distinct_owners=len(candidates), cohorts={k:dict(x) for k,x in report.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "sidecar", "raw", "trace", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    owners, papers = inventory(args.dataset, args.sidecar, args.raw)
    trace = json.loads(args.trace.read_bytes())
    report = analyze(owners, papers, trace)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
