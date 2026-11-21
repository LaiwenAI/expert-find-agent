"""Audit public enterprise sampling frames without running expert search."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from subprocess import check_output
from xml.etree import ElementTree
from zipfile import ZipFile

from bs4 import BeautifulSoup
from openpyxl import load_workbook

from experiments.freeze_enterprise_panel import digest_file, issuer_key, screen

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
BATCHES = ("second", "third", "fourth", "fifth", "sixth")
COUNTS = (70, 50, 50, 50, 50)
ORG = "\u5355\u4f4d\u540d\u79f0"
TECHNICAL = "\u6280\u672f\u521b\u65b0\u9700\u6c42\u60c5\u51b5\u8bf4\u660e"
COOPERATION = "\u4ea7\u5b66\u7814\u5408\u4f5c\u8981\u6c42"
NT_ORG = "\u4f01\u4e1a\u540d\u79f0"
NT_TITLE = "\u9879\u76ee\u540d\u79f0"
NT_BODY = "\u6280\u672f\u96be\u9898"
BOOK_FIELDS = ("\u547d\u9898\u540d\u79f0", "\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42", "\u547d\u9898\u4f01\u4e1a")
UNAVAILABLE = ("\u9690\u53bb", "\u9690\u85cf", "\u533f\u540d", "\u4fdd\u5bc6", "\u4e0d\u516c\u5f00", "\u672a\u516c\u5f00", "\u67d0\u516c\u53f8")


def listed_names(path):
    html = BeautifulSoup(path.read_bytes(), "html.parser")
    pairs = []
    for row in html.select("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td", recursive=False)]
        if len(cells) == 3 and cells[0].isdigit() and cells[1]:
            pairs.append((int(cells[0]), cells[1]))
    if len(dict(pairs)) != len(pairs):
        raise ValueError("Duplicate listed request ID")
    return dict(pairs)


def is_detail(filename):
    name = PurePosixPath(filename).name
    return name.lower().endswith(".docx") and not name.startswith("~$")


def attachments(path):
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            for member in archive.namelist():
                if is_detail(member):
                    yield member, archive.read(member)
    else:
        for member in check_output(["bsdtar", "-tf", str(path)], text=True).splitlines():
            if is_detail(member):
                yield member, check_output(["bsdtar", "-xOf", str(path), "--", member])


def parse_detail(data):
    with ZipFile(BytesIO(data)) as document:
        root = ElementTree.fromstring(document.read("word/document.xml"))
    rows = [["\n".join("".join(t.text or "" for t in p.findall(".//w:t", NS))
                      for p in cell.findall("./w:p", NS)).strip()
             for cell in row.findall("./w:tc", NS)] for row in root.findall(".//w:tr", NS)]
    names = [row[-1] for row in rows if len(row) > 1 and row[0] == ORG]
    start = next((i for i, row in enumerate(rows) if row and row[0] == TECHNICAL), None)
    stop = next((i for i, row in enumerate(rows) if row and row[0] == COOPERATION), None)
    if len(names) != 1 or start is None or stop is None or stop <= start:
        raise ValueError("Unexpected enterprise attachment schema")
    body = "\n".join(row[-1] for row in rows[start:stop] if len(row) > 1 and row[-1])
    return names[0], body


def package(root, batch, expectation):
    notice = root / f"xian-{batch}.html"
    listed = listed_names(notice)
    candidates = [p for p in root.glob(f"xian-{batch}.*") if p.suffix in (".zip", ".rar")]
    if len(candidates) != 1 or len(listed) != expectation:
        raise ValueError("Expected exactly one archive and the official listing count")
    archive = candidates[0]
    rows = []
    joined = set()
    for filename, content in attachments(archive):
        match = re.match(r"[0-9]+", PurePosixPath(filename).name)
        if match is None:
            raise ValueError("Attachment filename lacks its notice's request ID")
        number = int(match[0])
        if number not in listed or number in joined:
            raise ValueError("Attachment does not join uniquely to the notice")
        joined.add(number)
        org, body = parse_detail(content)
        rows.append(row(f"xian-{number}", batch, listed[number], body, org, source_row=number))
    if joined != listed.keys():
        raise ValueError("Notice and attachment inventories differ")
    return rows, {"notice_sha256": digest_file(notice), "attachment_sha256": digest_file(archive)}


def row(key, batch, title, body, organization, source_row):
    masked = not issuer_key(organization) or organization == "-" or any(word in organization for word in UNAVAILABLE)
    return dict(id=key, batch=batch, zh=f"{title}\n{body}", company="" if masked else organization,
                url="", group_id=None, source_row=source_row, split="sealed",
                complete=bool(body.strip()))


def nantong(path):
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    out = []
    for table in soup.select("table"):
        fields = {}
        for tr in table.select("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
            if len(cells) >= 2:
                fields[cells[0]] = cells[1]
        body = [value for key, value in fields.items() if key.startswith(NT_BODY)]
        if NT_ORG in fields and NT_TITLE in fields and len(body) == 1:
            index = len(out) + 1
            out.append(row(f"nantong-{index}", "nantong", fields[NT_TITLE], body[0],
                           fields[NT_ORG], source_row=index))
    if len(out) != 4:
        raise ValueError("Unexpected Nantong table count")
    return out


def historical(workbook):
    book = load_workbook(workbook, read_only=True, data_only=True)
    prior = []
    try:
        cells = iter(book.active.values)
        header = next(cells)
        indices = [header.index(field) for field in BOOK_FIELDS]
        for index, values in enumerate(cells, 2):
            title, body, organization = (str(values[i] or "") for i in indices)
            if title.strip() and body.strip():
                record = row(f"prior-{index}", "prior", title, body, organization, source_row=index)
                record["split"] = "prior"
                prior.append(record)
    finally:
        book.close()
    return prior


def main(args):
    if args.output.exists() or args.private_index.exists():
        raise FileExistsError("Require fresh output paths")
    candidates, sources = [], {}
    for batch, count in zip(BATCHES, COUNTS, strict=True):
        rows, sources[batch] = package(args.snapshot, batch, count)
        candidates.extend(rows)
    path = args.snapshot / "nantong.html"
    candidates.extend(nantong(path))
    sources["nantong"] = {"notice_sha256": digest_file(path)}
    usable = [r for r in candidates if r["complete"] and r["company"]]
    history = historical(args.workbook)
    selected, comparison = screen(usable + history, set(), set())
    by_id = {r["id"]: r for r in candidates}
    result = {
        "sources": sources,
        "listed": dict(Counter(r["batch"] for r in candidates)),
        "detailed": dict(Counter(r["batch"] for r in candidates if r["complete"])),
        "organization_present": dict(Counter(r["batch"] for r in candidates if r["company"])),
        "usable_distinct_normalized_organizations": len({issuer_key(r["company"]) for r in usable}),
        "historical_workbook_records_checked": len(history),
        "selected_by_source": dict(Counter(by_id[r["id"]]["batch"] for r in selected)),
        "screen": {key: value for key, value in comparison.items() if key != "original_sealed_groups"},
        "expert_queries_run": 0,
        "human_judgments": 0,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    private = {"records": selected, "historical_workbook_sha256": digest_file(args.workbook),
               "publication_sources": sources}
    args.private_index.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with args.private_index.open("x", encoding="utf-8") as stream:
        args.private_index.chmod(0o600)
        stream.write(json.dumps(private, indent=2) + "\n")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "workbook", "output", "private_index"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
