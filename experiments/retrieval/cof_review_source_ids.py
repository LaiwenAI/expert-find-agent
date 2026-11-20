"""Locate blinded CoF references by their public work IDs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from experiments.retrieval.cof_profile_depth_review import read_form
from experiments.retrieval.cof_public_provenance_audit import canonical
from experiments.retrieval.cof_rank_disagreement_review import sha, write_json
from experiments.retrieval.cof_review_references import lookup_cache, markdown

PANELS = {"kdd": "KDD", "scirepeval": "SciRepEval"}


def fetch(url, payload=None):
    request = Request(url, data=json.dumps(payload).encode() if payload else None,
                      headers={"Content-Type": "application/json",
                               "User-Agent": "ExpertSearchResearch/0.1 (public work lookup)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=18) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            wait = error.headers.get("Retry-After", "")
            seconds = min(60, int(wait)) if wait.isdecimal() else 2 ** attempt
        except (URLError, TimeoutError):
            seconds = 2 ** attempt
        if attempt < 2:
            sleep(seconds)
    raise RuntimeError(f"Could not query {urlparse(url).hostname}")


def retrieve(panel, ids):
    if panel == "kdd":
        fields = "id,doi,display_name,authorships,best_oa_location"
        for offset in range(0, len(ids), 40):
            chunk = ids[offset:offset+40]
            if any(re.fullmatch(r"[0-9]+", work) is None for work in chunk):
                raise ValueError("Unexpected OpenAlex source ID")
            query = urlencode({"filter": "openalex:" + "|".join("W" + work for work in chunk),
                               "select": fields, "per-page": len(chunk)})
            response = fetch("https://api.openalex.org/works?" + query)
            for item in response["results"]:
                identifier = item["id"].rsplit("/W", 1)[-1]
                authors = {x["author"]["id"].rsplit("/", 1)[-1] for x in item["authorships"]
                           if x["author"].get("id")}
                location = item.get("best_oa_location") or {}
                doi = (item.get("doi") or "").removeprefix("https://doi.org/")
                yield identifier, dict(title=item["display_name"], doi=doi, authors=sorted(authors),
                                       link=location.get("pdf_url") or location.get("landing_page_url"))
            sleep(.25)
    else:
        fields = urlencode({"fields": "paperId,title,externalIds,openAccessPdf,authors"})
        for offset in range(0, len(ids), 200):
            chunk = ids[offset:offset+200]
            if any(re.fullmatch(r"[0-9a-f]{40}", work) is None for work in chunk):
                raise ValueError("Unexpected Semantic Scholar source ID")
            response = fetch("https://api.semanticscholar.org/graph/v1/paper/batch?" + fields, {"ids": chunk})
            for requested, item in zip(chunk, response, strict=True):
                if item and requested == item["paperId"]:
                    yield requested, dict(
                        title=item["title"], doi=(item.get("externalIds") or {}).get("DOI"),
                        authors=sorted({row["authorId"] for row in item["authors"] if row.get("authorId")}),
                        link=(item.get("openAccessPdf") or {}).get("url"),
                    )
            sleep(.5)


def original_ids(panel, originals, packet):
    manifest = json.loads((packet / "manifest.json").read_bytes())
    form, case_map = packet / "review.jsonl", packet / "custodian" / "cases.json"
    if sha(form) != manifest["form_sha256"] or sha(case_map) != manifest["mapping_sha256"]:
        raise ValueError("The blind packet changed")
    rows = read_form(form)
    cases = json.loads(case_map.read_bytes())
    root = originals / PANELS[panel]
    with (root / f"{PANELS[panel]}_reviewers_test.json").open() as stream:
        owners = {row["reviewer"]: row["papers"] for row in map(json.loads, stream)}
    needed = {work for person in cases for work in owners[person["owner"]]}
    with (root / f"{PANELS[panel]}_papers_test.json").open() as stream:
        titles = {row["paper"]: row["title"] for row in map(json.loads, stream) if row["paper"] in needed}
    assembled = []
    for person in cases:
        anchor_titles = rows[person["sources"][0]["case"]]["profile_titles"]
        ids = {canonical(title): [] for title in anchor_titles}
        for work in dict.fromkeys(owners[person["owner"]]):
            title = canonical(titles[work])
            if title in ids:
                ids[title].append(work)
        checked = {}
        for source in person["sources"]:
            work, case = source["source"], source["case"]
            if work not in owners[person["owner"]] or titles[work] != rows[case]["citation"].partition("\n")[0]:
                raise ValueError("Citation and catalog record disagree")
            checked[case] = work
        assembled.append((person["owner"], checked, ids))
    if {case for _, sources, _ in assembled for case in sources} != rows.keys():
        raise ValueError("Missing case records")
    return rows, assembled, titles


def link(url):
    parsed = urlparse(url or "")
    if parsed.scheme == "https" and parsed.hostname:
        return quote(url, safe=":/?&=%#;+@-._~")
    return ""


def review_panel(panel, packet, originals, cache, output, *, offline, title_records):
    forms, assembled, original_titles = original_ids(panel, originals, packet)
    sources = {work for _, docs, anchors in assembled for work in
               (*docs.values(), *(i for values in anchors.values() for i in values))}
    records = json.loads(cache.read_bytes()) if cache.exists() else {}
    if not offline:
        records.update(retrieve(panel, sorted(sources - records.keys())))
        write_json(cache, records)
    accepted = {work: item for work, item in records.items()
                if work in sources and item.get("title") and canonical(item["title"]) == canonical(original_titles[work])}
    summary = Counter(profiles=len(assembled), requested_work_ids=len(sources), metadata_returned=len(sources & records.keys()),
                      exact_title_work_ids=len(accepted), citation_cases=len(forms))
    sheets = {}
    for owner, cases, anchor_sources in assembled:
        # Work versions with the same normalized title are not independent
        # identity anchors. Reject title groups with ambiguous owner IDs.
        slots = {}
        for group in anchor_sources.values():
            if len(group) == 1 and group[0] in accepted:
                work = group[0]
                item = accepted[work]
                slots.setdefault((item["doi"] or work).casefold(), (work, item))
        authors = [set(x["authors"]) for _, x in slots.values() if x["authors"]]
        candidates = set.intersection(*authors) if len(authors) >= 2 else set()
        identity = next(iter(candidates)) if len(candidates) == 1 else None
        summary["profiles_with_two_anchor_bylines"] += len(authors) >= 2
        summary["profiles_with_unique_shared_registry_author"] += identity is not None
        summary["profiles_matching_hypothesized_openalex_author_id"] += panel == "kdd" and identity == f"A{owner}"
        for case, work in cases.items():
            outcome = accepted.get(work)
            lines = [f"## {case}", "", f"- Citation · {markdown(original_titles[work])}"]
            site = ("https://openalex.org/W" if panel == "kdd"
                    else "https://www.semanticscholar.org/paper/")
            if not outcome:
                lines.append("  Work ID could not be verified.")
            if title_records is not None:
                title_result = title_records.get(original_titles[work], {})
                if title_result.get("status") == "exact":
                    title_doi = title_result["doi"]
                    if not outcome or not outcome["doi"]:
                        label = "Title-only DOI lead"
                    elif outcome["doi"].casefold() != title_doi.casefold():
                        label = "Conflicting title-lookup DOI"
                    else:
                        label = None
                    if label:
                        lines.append(f"  [{label}](https://doi.org/{quote(title_doi, safe='/')})")
            for identifier, entry in [(work, outcome), *slots.values()]:
                if not entry:
                    continue
                if identifier != work:
                    lines.append(f"- Profile paper · {markdown(entry['title'])}")
                lines.append(f"  [Registry work](<{site}{identifier}>)")
                if entry["doi"]:
                    lines.append(f"  [Publisher DOI](https://doi.org/{quote(entry['doi'], safe='/')})")
                if url := link(entry["link"]):
                    lines.append(f"  [Open-access copy]({url})")
            sheets[case] = "\n".join(lines)
            summary["cited_cases_resolved_by_id"] += outcome is not None
            summary["cited_cases_with_doi"] += bool(outcome and outcome["doi"])
            if title_records is not None:
                title_result = title_records.get(original_titles[work], {})
                status = title_result.get("status", "not_checked")
                summary[f"title_lookup_{status}"] += 1
                if status == "exact" and outcome and outcome["doi"]:
                    summary["cases_with_both_dois"] += 1
                    summary["cases_with_disagreeing_dois"] += (
                        outcome["doi"].casefold() != title_result["doi"].casefold()
                    )
            summary["cited_cases_with_open_access_link"] += bool(outcome and link(outcome["link"]))
            summary["cited_cases_with_anchor_identity_on_byline"] += bool(
                identity and outcome and identity in outcome["authors"])
    (output / f"{panel}.md").write_text(
        "# Source links for blind review\n\n"
        "Use the case code to locate the immutable review row. Confirm the work and any author contribution in the original.\n\n"
        + "\n\n".join(sheets[case] for case in forms) + "\n"
    )
    return dict(summary, metadata_cache_sha256=sha(cache))


def run(packet, originals, cache, output, offline=False, title_cache=None):
    if output.exists():
        raise ValueError("Choose a fresh reference directory")
    if offline and any(not (cache / f"{panel}.json").is_file() for panel in PANELS):
        raise ValueError("Offline rendering requires both cached panels")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    records, title_cache_digest = lookup_cache(title_cache) if title_cache else (None, None)
    panels = {panel: review_panel(panel, packet / panel, originals, cache / f"{panel}.json", output,
                                  offline=offline, title_records=records) for panel in PANELS}
    write_json(output / "summary.json", dict(script_sha256=sha(Path(__file__)), panels=panels,
                                             title_cache_sha256=title_cache_digest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "originals", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--title-cache", type=Path)
    run(**vars(parser.parse_args()))
