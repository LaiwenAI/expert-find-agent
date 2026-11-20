"""Reconstruct audited condition-to-PDF-page pairs with shared reviewer selections."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse

from experiments.credit.crosspage_pdf_panel import REVISION, SOURCE_HOST, load, materialize, write


MAX_BYTES = 15_000_000
EXCLUDED_CASES = {"J011", "J060", "J061"}
ADJUDICATION = Path("audit/admin/final_adjudication/three_annotator_adjudication_results.json")


def metadata(root):
    review = json.loads((root / ADJUDICATION).read_bytes())
    pages = {row["doc_id"]: row["page_count"] for row in load(root / "data/v1/page_counts.jsonl")}
    sources = {
        row["doc_id"]: row | {"page_count": pages[row["doc_id"]]}
        for row in load(root / "data/v1/corpus.jsonl")
        if row["source"] == "world_bank" and row.get("sha256") and row["doc_id"] in pages
        and row["bytes"] <= MAX_BYTES and urlparse(row["pdf_url"]).scheme == "https"
        and urlparse(row["pdf_url"]).hostname == SOURCE_HOST
    }
    conditions = []
    for item in review["items"]:
        doc = item["doc_id"]
        if item["release"] != "v1" or doc not in sources or item["blind_id"] in EXCLUDED_CASES:
            continue
        for criterion in item["conditions"]:
            accepted = sorted(set(criterion["selected_pages"]["A"]) & set(criterion["selected_pages"]["E"]))
            if criterion["recommended_support"] and accepted:
                if min(accepted) < 1 or max(accepted) > pages[doc]:
                    raise ValueError("Audited page outside the original PDF")
                conditions.append(dict(
                    case=item["blind_id"], query_id=item["query_id"], doc=doc,
                    condition_index=criterion["source_condition_index"], condition=criterion["condition"],
                    accepted_pages=accepted, family=sources[doc]["family_id"],
                ))
    return conditions, {doc: sources[doc] for doc in sorted({row["doc"] for row in conditions})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "cache", "output", "manifest"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise ValueError("Use fresh panel and manifest paths")
    args.cache.mkdir(parents=True, exist_ok=True)
    rows, sources = metadata(args.root)
    upstream_hashes = {
        str(path): sha256((args.root / path).read_bytes()).hexdigest()
        for path in (ADJUDICATION, Path("data/v1/corpus.jsonl"), Path("data/v1/page_counts.jsonl"))
    }
    pdf_hashes = {doc: data["sha256"] for doc, data in sources.items()}
    frozen = json.loads(args.frozen.read_bytes()) if args.frozen else None
    if frozen:
        if (frozen["upstream_revision"] != REVISION or frozen["upstream_sha256"] != upstream_hashes
                or frozen["pdf_hashes"] != pdf_hashes):
            raise ValueError("Source metadata differs from the frozen archive")
        sources = {doc: source for doc, source in sources.items() if doc not in frozen["unavailable"]}
    recover = partial(materialize, cache=args.cache, max_bytes=MAX_BYTES)
    documents = {}
    unavailable = {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        for doc, payload, error in workers.map(recover, sources.items()):
            if error:
                unavailable[doc] = error
            else:
                documents[doc] = payload
    retained = [row for row in rows if row["doc"] in documents]
    if not retained:
        raise ValueError("No recoverable audited rows")
    selected = [(row["case"], row["condition_index"]) for row in retained]
    if frozen and (unavailable or [list(row) for row in selected] != frozen["selected_rows"]):
        raise ValueError("Cannot restore all frozen study conditions")
    write(args.output, dict(rows=retained, documents=documents))
    panel_hash = sha256(args.output.read_bytes()).hexdigest()
    if frozen and panel_hash != frozen["panel_sha256"]:
        raise ValueError("Restored text differs from the frozen archive")
    write(args.manifest, frozen or dict(
        upstream_revision=REVISION, upstream_sha256=upstream_hashes,
        panel_sha256=panel_hash, pdf_hashes=pdf_hashes, selected_rows=selected,
        unavailable=unavailable,
    ))
    print(f"Retained {len(retained)}/{len(rows)} reviewer-agreed conditions in "
          f"{len(documents)}/{len(sources)} PDFs.", flush=True)


if __name__ == "__main__":
    main()
