"""Verify PDF availability and explicit contribution cues for public review links."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from html.parser import HTMLParser

from experiments.retrieval.cof_review_source_ids import canonical, link, original_ids
from experiments.retrieval.cof_rank_disagreement_review import sha, write_json

HOSTS = {
    "arxiv.org", "export.arxiv.org", "www.ijcai.org", "opus.lib.uts.edu.au",
    "dl.acm.org", "doi.org", "ieeexplore.ieee.org", "hal.science",
    "dr.ntu.edu.sg", "qmro.qmul.ac.uk", "inria.hal.science",
    "hal.archives-ouvertes.fr", "discovery.ucl.ac.uk", "www.repository.cam.ac.uk",
    "ijcai.org", "ojs.aaai.org", "cf002.cdn.4science.cloud",
}
MAX_BYTES = 18_000_000
WORKERS = 6
CONTRIBUTION_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.]?\s+)?(?:"
    r"(?:credit\s+)?authors?(?:hip)?(?:['’]s?)?\s+contributions?(?:\s+statement)?"
    r"|contributions?\s+of\s+(?:the\s+)?authors?"
    r"|contributorship(?:\s+statement)?"
    r")\s*[:.]?\s*$",
    re.I,
)
EQUAL_CONTRIBUTION = re.compile(r"\bcontributed equally\b", re.I)


def allowed(url):
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in HOSTS and not parsed.username


class PublicRedirect(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, request, fp, code, msg, headers, new_url):
        if not allowed(new_url):
            raise ValueError("redirect_outside_sample_hosts")
        return super().redirect_request(request, fp, code, msg, headers, new_url)


class CitationMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdf_url = None

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            fields = dict(attrs)
            if fields.get("name", "").casefold() == "citation_pdf_url":
                self.pdf_url = fields.get("content")


def download(url):
    if not allowed(url):
        return None, "outside_sample_hosts"
    headers = {"User-Agent": "ExpertSearchResearch/0.1 (public publication provenance)"}
    try:
        with build_opener(PublicRedirect()).open(Request(url, headers=headers), timeout=12) as response:
            data = response.read(MAX_BYTES+1)
            final_url = response.geturl()
    except HTTPError as error:
        return None, f"http_{error.code}"
    except (URLError, TimeoutError):
        return None, "network_error"
    except ValueError as error:
        return None, str(error)
    if len(data) > MAX_BYTES:
        return None, "download_limit"
    return (final_url, data), None


def pdf_text(data):
    result = subprocess.run(
        ["pdftotext", "-layout", "-", "-"], input=data, capture_output=True,
        timeout=15, check=False,
    )
    if result.returncode:
        return None
    return result.stdout.decode(errors="replace")


def inspect(url, titles):
    visited = set()
    for _ in range(3):
        if url in visited:
            return {"status": "pdf_discovery_loop"}
        visited.add(url)
        response, error = download(url)
        if error:
            return {"status": error}
        final_url, data = response
        if b"%PDF-" in data[:1024]:
            try:
                text = pdf_text(data)
            except subprocess.TimeoutExpired:
                return {"status": "pdf_extraction_timeout"}
            if not text or len(canonical(text)) < 2_000:
                return {"status": "pdf_no_text"}
            pages = text.split("\f")
            front = canonical(" ".join(pages[:3]))
            seen = [title for title in titles if f" {canonical(title)} " in f" {front} "]
            headings, equal = [], []
            for n, page in enumerate(pages, 1):
                for line in page.splitlines():
                    if CONTRIBUTION_HEADING.fullmatch(line.strip()):
                        headings.append(n)
                if EQUAL_CONTRIBUTION.search(" ".join(page.split())):
                    equal.append(n)
            return {"status": "title_verified_pdf" if seen else "pdf_title_unverified", "titles": seen,
                    "final_url": final_url, "heading_pages": sorted(set(headings)),
                    "equal_contribution_marker_pages": equal}
        parser = CitationMetadata()
        parser.feed(data.decode(errors="replace"))
        if not parser.pdf_url:
            return {"status": "landing_page_without_pdf_metadata"}
        url = urljoin(final_url, parser.pdf_url)
    return {"status": "pdf_discovery_limit"}


def inspect_with_versions(url, titles):
    original = inspect(url, titles)
    verified = original.get("final_url", "")
    target = urlparse(verified)
    paper = re.fullmatch(r"/pdf/(\d{4}\.\d{4,5})(?:\.pdf)?", target.path)
    if original["status"] != "pdf_title_unverified" or target.hostname != "arxiv.org" or not paper:
        return original, original, []
    attempts = []
    for version in (1, 2):
        candidate = f"https://arxiv.org/pdf/{paper[1]}v{version}"
        outcome = inspect(candidate, titles)
        attempts.append((version, outcome["status"]))
        if outcome["status"] == "title_verified_pdf":
            return original, outcome, attempts
    return original, original, attempts


def by_url(original, packet, registry):
    statuses = {}
    for panel in ("kdd", "scirepeval"):
        _, papers, titles = original_ids(panel, original, packet / panel)
        records = json.loads((registry / f"{panel}.json").read_bytes())
        for _, sources, _ in papers:
            for case, doc in sources.items():
                record = records.get(doc)
                if record and canonical(record["title"]) == canonical(titles[doc]):
                    url = link(record["link"])
                    if url:
                        statuses.setdefault(url, []).append((panel, case, titles[doc]))
    return statuses


def run(original, packet, registry, private, output):
    if shutil.which("pdftotext") is None or output.exists() or private.exists():
        raise ValueError("Require Poppler and fresh audit paths")
    cases = by_url(original, packet, registry)
    if (len(cases), sum(map(len, cases.values()))) != (29, 31):
        raise ValueError("Changed public-link audit cohort")
    outcomes = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        tasks = {url: pool.submit(inspect_with_versions, url, sorted({title for _, _, title in rows}))
                 for url, rows in cases.items()}
        for url, future in tasks.items():
            outcomes[url] = future.result()
    initial_counts = Counter(before["status"] for before, _, _ in outcomes.values())
    counts = Counter(after["status"] for _, after, _ in outcomes.values())
    match_counts = Counter()
    review = []
    for url, (before, result, attempts) in outcomes.items():
        matched = set(result.get("titles", []))
        for panel, case, title in cases[url]:
            resolved = title in matched
            match_counts["title_verified_pdf_cases"] += resolved
            match_counts["case_with_contribution_heading"] += resolved and bool(result["heading_pages"])
            match_counts["case_with_equal_contribution_marker"] += resolved and bool(
                result["equal_contribution_marker_pages"])
            match_counts["cases_recovered_by_version_lookup"] += resolved and bool(attempts)
            review.append(dict(panel=panel, case=case, url=url, title=title, initial_status=before["status"],
                               version_attempts=attempts, outcome=result))
    write_json(private, review)
    private.chmod(0o600)
    write_json(output, dict(unique_urls=len(cases), linked_citation_cases=len(review),
                            initial_access_status_counts=dict(initial_counts),
                            access_status_counts=dict(counts), case_counts=dict(match_counts),
                            version_attempts=sum(len(attempts) for _, _, attempts in outcomes.values()),
                            script_sha256=sha(Path(__file__)), private_result_sha256=sha(private),
                            registry_hashes={panel: sha(registry/f"{panel}.json") for panel in ("kdd", "scirepeval")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "packet", "registry", "private", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
