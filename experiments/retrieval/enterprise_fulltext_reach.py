"""Check saved public paper pointers and locate declared author contributions."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree as ET

from experiments.retrieval.enterprise_paper_identity_audit import digest, key, names, record_hash


PUBLISHERS = {
    "dl.acm.org", "fec.bjfu.edu.cn", "files.sciengine.com", "geophy.cn",
    "ieeevtc.org", "ieeexplore.ieee.org", "journal_xdxb.xidian.edu.cn",
    "link.springer.com", "ope.lightpublishing.cn", "pdf.hanspub.org", "qikan.cmes.org",
    "wulixb.iphy.ac.cn", "www.fxcsxb.com", "www.kjjb.org", "www.researching.cn",
    "www.sciengine.com", "www.sciopen.com", "www.tandfonline.com", "xtglxb.sjtu.edu.cn",
    "journals.plos.org",
}
MAX_PDF = 16 * 1024 * 1024
CONTROL = "https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0244611&type=printable"
CONTROL_TITLE = "Documenting contributions to scholarly articles using CRediT and tenzing"
SECTION = re.compile(
    r"^\s*(?:author(?:s['’]?|['’]s)?\s+contributions?(?:\s+statement)?|"
    r"credit\s+authorship(?:\s+contribution)?\s+statement|"
    r"contributions?\s+of\s+(?:the\s+)?authors|作\s*者\s*贡\s*献(?:声明|说明)?|作\s*者\s*分\s*工)"
    r"(?=\W|$)",
    re.IGNORECASE,
)
SECTION_END = re.compile(
    r"^\s*(?:(?:declaration of\s+)?competing interests?|conflicts? of interest|"
    r"references|bibliography|acknowledge?ments?|funding|data availability|"
    r"supporting information|supplementary materials?|参考文献|致谢|利益冲突)\s*[:：.]?\s*$",
    re.IGNORECASE,
)
DISCLOSURE_MENTION = re.compile(
    r"author(?:s['’]?|['’]s)?\s+contributions?|credit\s+authorship|作\s*者\s*贡\s*献|作\s*者\s*分\s*工",
    re.IGNORECASE,
)
ABSTRACT_HEAD = re.compile(r"^\s*(?:abstract|摘\s*要)\s*(?:[:：.]|$)", re.IGNORECASE | re.MULTILINE)
DOI = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


def allowed(url):
    try:
        parts = urlsplit(url)
        if not (
            parts.scheme == "https" and parts.hostname in PUBLISHERS and parts.port in {None, 443}
            and not parts.username and not parts.password and parts.path.startswith("/")
        ):
            return None
        fields = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
        for field, value in fields:
            if field in {"id", "attachType"} and re.fullmatch(r"[0-9]{1,12}", value):
                continue
            if parts.hostname == "journals.plos.org" and (
                field == "id" and bool(DOI.fullmatch(value)) or field == "type" and value == "printable"
            ):
                continue
            if field == "id" and re.fullmatch(r"[A-Za-z0-9-]{24,36}", value):
                continue
            if field == "attachType" and value.casefold() in {"pdf", "fulltext"}:
                continue
            return None
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    except ValueError:
        return None


class AllowedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        verified = allowed(newurl)
        if not verified:
            raise URLError("redirect_outside_scope")
        return super().redirect_request(req, fp, code, msg, headers, verified)


def capture(task):
    url, folder = task
    fingerprint = sha256(url.encode()).hexdigest()
    metadata = folder / f"{fingerprint}.json"
    pdf = folder / f"{fingerprint}.pdf"
    if metadata.exists():
        saved = json.loads(metadata.read_bytes())
        if saved["status"] == "pdf" and digest(pdf) != saved["sha256"]:
            raise ValueError("Cached PDF changed")
        return url, saved["status"], pdf
    if allowed(url) != url:
        raise ValueError("Capture requires an approved anonymous URL")
    request = Request(url, headers={"User-Agent": "Academic source reach audit", "Accept": "application/pdf,*/*;q=0.5"})
    status = "network_error"
    saved = {}
    try:
        with build_opener(AllowedRedirect()).open(request, timeout=10) as response:
            first = response.read(1024)
            if b"%PDF-" not in first:
                status = "non_pdf_response"
            else:
                content = first + response.read(MAX_PDF - len(first) + 1)
                if len(content) > MAX_PDF:
                    status = "size_limit"
                else:
                    pdf.write_bytes(content)
                    status = "pdf"
                    saved["sha256"] = digest(pdf)
    except HTTPError as error:
        status = f"http_{error.code}"
    except (URLError, TimeoutError, OSError) as error:
        if isinstance(error, URLError) and str(error.reason) == "redirect_outside_scope":
            status = "redirect_outside_scope"
    metadata.write_text(json.dumps(dict(saved, status=status), sort_keys=True) + "\n")
    return url, status, pdf


def load_panel(dataset, sidecar, raw_path, trace_path):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    meta = json.loads((sidecar / "manifest.json").read_bytes())
    for path, checksum in (
        (dataset / "experts.jsonl", manifest["files_sha256"]["experts.jsonl"]),
        (sidecar / "papers.jsonl", meta["metadata_sha256"]),
        (raw_path, meta["source_sha256"]),
    ):
        if digest(path) != checksum:
            raise ValueError(f"Changed input: {path.name}")
    trace = json.loads(trace_path.read_bytes())["observations"]
    if len(trace) != 80 or len({row["id"] for row in trace}) != 80:
        raise ValueError("Expected the frozen 80-request development trace")
    requests = []
    for item in trace:
        arm = item["arms"]["hybrid_prefixes"]
        if arm["issued"]:
            continue
        requests.append({source for views in arm["plan"] for source, _, _ in views if source.startswith("paper:")})
    targets = set().union(*requests)
    with (sidecar / "papers.jsonl").open() as stream:
        rows = {row["raw_line"]: row for line in stream if (row := json.loads(line))["source_id"] in targets}
    if {row["source_id"] for row in rows.values()} != targets:
        raise ValueError("Some selected paper sources are absent from the archive metadata")
    required = {row["expert_id"] for row in rows.values()}
    with (dataset / "experts.jsonl").open() as stream:
        owners = {row["id"]: names(row["name"]) for line in stream
                  if (row := json.loads(line))["id"] in required and row["split"] == "development"}
    papers = {}
    with raw_path.open() as stream:
        for number, line in enumerate(stream, 1):
            if number not in rows:
                continue
            meta_row = rows[number]
            raw = json.loads(line)
            if record_hash(raw) != meta_row["raw_record_sha256"]:
                raise ValueError("Paper and evidence locator out of alignment")
            papers[meta_row["source_id"]] = dict(
                title=raw["title"], alt_title=raw.get("gs_title") or "", doi=(meta_row["doi"] or "").casefold(),
                owner_names=owners[meta_row["expert_id"]], raw_pointer=raw.get("gs_pdf_link") or "",
            )
    return papers, requests


def open_pdf(path, limit=None, *, layout="-layout"):
    page_range = ["-f", "1", "-l", str(limit)] if limit else []
    result = subprocess.run(
        ["pdftotext", layout, "-enc", "UTF-8", *page_range, str(path), "-"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=20, check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def title_match(text, expected):
    normalized = key(expected)
    return len(normalized) >= 8 and normalized in key(text)


def statement_windows(text):
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        heading = SECTION.search(line)
        if not heading:
            continue
        statement = [line[heading.end():]]
        for next_line in lines[i + 1:i + 25]:
            if SECTION_END.fullmatch(next_line) or SECTION.search(next_line):
                break
            if next_line.strip():
                statement.append(next_line.strip())
        result.append(" ".join(statement).strip())
    return result


def examine_pdf(path, paper):
    first = open_pdf(path, 2)
    first_title = title_match(first, paper["title"])
    alt_title = title_match(first, paper["alt_title"])
    heading = ABSTRACT_HEAD.search(first)
    byline_region = key(first[:heading.start()]) if heading else ""
    candidate_version = first_title or alt_title
    pages = open_pdf(path) if candidate_version else ""
    layout_windows = statement_windows(pages)
    raw_windows = statement_windows(open_pdf(path, layout="-raw")) if candidate_version and not layout_windows else []
    named_windows = layout_windows or raw_windows
    original, aliases = paper["owner_names"]
    doi_text = {v.rstrip(".;),").casefold() for v in DOI.findall(first)}
    return dict(
        front_text=bool(first.strip()), primary_title=first_title, alternative_title=alt_title,
        stored_doi_in_front=bool(paper["doi"]) and paper["doi"] in doi_text,
        title_either=first_title or alt_title,
        candidate_name_before_abstract=bool(original and original in byline_region)
        or any(alias in byline_region for alias in aliases),
        disclosure_heading=bool(named_windows),
        disclosure_heading_layout_only=bool(layout_windows),
        disclosure_found_by_raw_fallback=bool(raw_windows),
        disclosure_mention=bool(DISCLOSURE_MENTION.search(pages)),
        exact_name_in_disclosure=bool(original) and any(original in key(window) for window in named_windows),
        phonetic_name_in_disclosure=any(alias in key(window) for window in named_windows for alias in aliases),
    )


def jats_records(folders):
    seen = set()
    for folder in folders:
        for path in sorted(folder.glob("*.xml")):
            raw = path.read_bytes()
            checksum = sha256(raw).digest()
            if checksum in seen:
                continue
            seen.add(checksum)
            root = ET.fromstring(raw)
            meta = root.find("./front/article-meta")
            if meta is None:
                continue
            title = meta.find("./title-group/article-title")
            doi = meta.findtext("article-id[@pub-id-type='doi']", "").casefold()
            note = root.findall(".//fn[@fn-type='con']")
            direct = meta.findall("contrib-group/contrib/role")
            notes = root.findall(".//sec[@sec-type='author-contributions']")
            yield "" if title is None else "".join(title.itertext()), doi, bool(note or direct or notes)


def search_jats(paper, archive):
    titles = {key(paper["title"]), key(paper["alt_title"])} - {""}
    rows = [(title, doi, roles) for title, doi, roles in archive if key(title) in titles]
    return dict(
        jats_title_match=bool(rows),
        jats_title_and_doi=bool(paper["doi"]) and any(paper["doi"] == doi for _, doi, _ in rows),
        jats_contribution_field=any(roles for _, _, roles in rows),
    )


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    papers, requests = load_panel(args.dataset, args.metadata, args.raw, args.trace)
    approved = [allowed(paper["raw_pointer"]) for paper in papers.values()]
    urls = sorted({url for url in approved if url} | {CONTROL})
    if args.offline and any(not (args.cache / (sha256(url.encode()).hexdigest() + ".json")).exists() for url in urls):
        raise ValueError("Replay requires a complete frozen capture cache")
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = {url: (status, path) for url, status, path in pool.map(capture, ((url, args.cache) for url in urls))}
    control_status, control_path = saved[CONTROL]
    control = examine_pdf(control_path, dict(title=CONTROL_TITLE, alt_title="", doi="", owner_names=("alexoholcombe", set()))) \
        if control_status == "pdf" else {"primary_title": False, "disclosure_heading": False}
    archive = list(jats_records(args.jats))
    outcomes = {}
    counts = Counter()
    for source, paper in papers.items():
        url = allowed(paper["raw_pointer"])
        status, path = saved[url] if url else ("outside_scope" if paper["raw_pointer"] else "no_pointer", None)
        record = dict(pointer=bool(paper["raw_pointer"]), attempted=bool(url), pdf=status == "pdf",
                      **search_jats(paper, archive))
        if status == "pdf":
            record.update(examine_pdf(path, paper))
        counts[status] += 1
        outcomes[source] = record
    paper_counts = Counter()
    for record in outcomes.values():
        paper_counts.update(record)
    coverage = Counter()
    for sources in requests:
        for flag in paper_counts:
            coverage[flag] += any(outcomes[source].get(flag, False) for source in sources)
    result = dict(
        requests=len(requests), paper_rows=len(papers), unique_attempted_urls=len(urls) - 1,
        paper_counts=dict(paper_counts), coverage=dict(coverage), status_counts=dict(counts),
        xml_records=len(archive), positive_control=dict(status=control_status, **control),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "metadata", "raw", "trace", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--jats", type=Path, nargs="*", default=())
    parser.add_argument("--offline", action="store_true")
    run(parser.parse_args())
