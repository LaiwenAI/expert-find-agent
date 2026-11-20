"""Freeze and reconstruct a bounded World Bank subset of n-Clue CrossPage."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlparse


LIMITS = {2: 16, 3: 12}
TRIALS = {2: 25, 3: 20}
MAX_BYTES = 6_000_000
MAX_PAGES = 90
SOURCE_HOST = "documents.worldbank.org"
REVISION = "8a19258214b2e6973cdc07614908c5caf654fd51"


def load(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(row) for row in source]


def hash_order(*parts):
    return sha256(":".join(("crosspage-local-text-v1", *map(str, parts))).encode()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False)
        stream.write("\n")


def choose(root):
    wanted = ("queries", "qrels", "corpus", "page_counts")
    data = {name: load(root / f"{name}.jsonl") for name in wanted}
    pages = {row["doc_id"]: row["page_count"] for row in data["page_counts"]}
    documents = {row["doc_id"]: row for row in data["corpus"]
                 if row["source"] == "world_bank" and row.get("sha256")
                 and urlparse(row["pdf_url"]).scheme == "https"
                 and urlparse(row["pdf_url"]).hostname == SOURCE_HOST
                 and row["bytes"] <= MAX_BYTES and 5 <= pages.get(row["doc_id"], 0) <= MAX_PAGES}
    grades = defaultdict(dict)
    for row in data["qrels"]:
        if row["doc_id"] in documents:
            grades[row["query_id"]][row["doc_id"]] = row["grade"]
    by_arity = defaultdict(list)
    for row in sorted(data["queries"], key=lambda q: hash_order(q["query_id"])):
        qid, n = row["query_id"], row["n"]
        if n not in LIMITS:
            continue
        ordered = sorted(grades[qid], key=lambda doc: hash_order(qid, doc))
        good = [doc for doc in ordered if grades[qid][doc] == n]
        partial = [doc for doc in ordered if 0 < grades[qid][doc] < n]
        if good and len(partial) >= 3 and len(by_arity[n]) < TRIALS[n]:
            # Pick only from official candidate labels. They never reach inference.
            group = []
            families = {documents[good[0]]["family_id"]}
            for doc in partial:
                family = documents[doc]["family_id"]
                if family not in families:
                    families.add(family)
                    group.append(doc)
                if len(group) == 3:
                    break
            if len(group) == 3:
                selected = {doc: grades[qid][doc] for doc in sorted([good[0], *group])}
                by_arity[n].append(dict(query_id=qid, arity=n, query=row["query_text_eval_full"],
                                        conditions=[item["concept"] for item in row["conditions"]],
                                        grades=selected))
    sources = {doc: documents[doc] | {"page_count": pages[doc]}
               for case in chain.from_iterable(by_arity.values()) for doc in case["grades"]}
    return dict(by_arity), sources, {name: sha256((root / f"{name}.jsonl").read_bytes()).hexdigest()
                                    for name in wanted}


def materialize(doc_info, cache, *, max_bytes=MAX_BYTES):
    doc, row = doc_info
    target = cache / f"{doc}.pdf"
    try:
        if not target.exists():
            with TemporaryDirectory(prefix="crosspage-pdf-", dir=cache) as scratch:
                temp = Path(scratch) / "download.pdf"
                result = subprocess.run(
                    ["curl", "-fsSL", "--max-time", "20", "--max-filesize", str(max_bytes),
                     row["pdf_url"], "-o", str(temp)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                )
                if result.returncode != 0:
                    return doc, None, f"download_{result.returncode}"
                if sha256(temp.read_bytes()).hexdigest() != row["sha256"]:
                    return doc, None, "hash_mismatch"
                temp.rename(target)
        if sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            return doc, None, "hash_mismatch"
        extracted = subprocess.check_output(["pdftotext", "-layout", str(target), "-"],
                                            stderr=subprocess.DEVNULL, timeout=20)
        pages = extracted.decode("utf-8").split("\f")
        if pages[-1].strip() == "":
            pages.pop()
        pages = [" ".join(page.split()) for page in pages]
        if len(pages) != row["page_count"] or sum(len(p) >= 120 for p in pages) < 3:
            return doc, None, "text_or_page_count"
        return doc, dict(title=row["title"], pages=pages), None
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return doc, None, "extraction_error"


def build(root, cache, output, manifest, *, frozen=None):
    cache.mkdir(parents=True, exist_ok=True)
    if output.exists() or manifest.exists():
        raise ValueError("Use fresh panel and manifest paths")
    cases, sources, provenance = choose(root)
    prior = json.loads(frozen.read_bytes()) if frozen else None
    if prior is not None:
        if prior["upstream_revision"] != REVISION or prior["upstream_sha256"] != provenance:
            raise ValueError("Source metadata differs from the frozen archive")
        queries = {row["query_id"]: row for row in chain.from_iterable(cases.values())}
        selected = list(prior["included_queries"])
        eligible = [queries[qid] for qid in selected]
        if any(row["grades"] != prior["included_queries"][row["query_id"]] for row in eligible):
            raise ValueError("Candidate pools differ from the frozen archive")
        sources = {doc: sources[doc] for doc in prior["expected_source_pdfs"]}
        if any(row["sha256"] != prior["expected_source_pdfs"][doc] for doc, row in sources.items()):
            raise ValueError("Changed source PDF hashes")
    with ThreadPoolExecutor(max_workers=6) as pool:
        docs = {}
        failed = {}
        for doc, text, error in pool.map(lambda item: materialize(item, cache), sorted(sources.items())):
            if error:
                failed[doc] = error
            else:
                docs[doc] = text
    if prior is not None:
        if failed:
            raise ValueError(f"Cannot reconstruct {len(failed)} required PDFs")
    else:
        eligible = [case for n, rows in sorted(cases.items())
                    for case in [row for row in rows if set(row["grades"]) <= docs.keys()][:LIMITS[n]]]
    if not eligible:
        raise ValueError(f"No eligible cases after reconstructing {len(docs)} documents")
    inventory = sorted({doc for row in eligible for doc in row["grades"]})
    write(output, dict(queries=eligible, documents={doc: docs[doc] for doc in inventory}))
    if prior is not None:
        if sha256(output.read_bytes()).hexdigest() != prior["panel_sha256"]:
            raise ValueError("Extracted text differs from the frozen panel")
        write(manifest, prior)
        print(f"Restored {len(eligible)} cases and {len(inventory)} PDFs", flush=True)
        return
    write(manifest, dict(
        upstream_revision=REVISION, upstream_sha256=provenance,
        panel_sha256=sha256(output.read_bytes()).hexdigest(),
        expected_source_pdfs={doc: sources[doc]["sha256"] for doc in inventory},
        included_queries={row["query_id"]: row["grades"] for row in eligible},
        intake_pdf_failures=failed, attempted=len(sources),
    ))
    print(f"{len(eligible)} cases and {len(inventory)} reconstructed PDFs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "cache", "output", "manifest"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--frozen", type=Path)
    build(**vars(parser.parse_args()))
